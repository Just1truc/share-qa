import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import datetime
import faiss
import ijson
import json
import sys
import csv
import re

from time import time
from typing import Tuple, TextIO
from pyarrow.parquet import ParquetFile
from torch import cuda

class OutputRow(dict):
    
    def __init__(
        self,
        dialog          : list[str],
        speakers        : list[str],
        positive_pairs  : list[int],
        QA              : dict | None = None,
        Δ_state         : dict | None = None
    ):
        super().__init__()
        losses = ["mlm", "cluster"]
        if QA != None:
            losses.append("qa")
        if Δ_state != None:
            losses.append("state")
            
        super().__setitem__("Dialog (EDUs)", json.dumps(dialog))
        super().__setitem__("Speakers", json.dumps(speakers))
        super().__setitem__("Positive Pairs", positive_pairs)
        super().__setitem__("QA", str(QA))
        super().__setitem__("ΔState", str(Δ_state))
        super().__setitem__("Losses", json.dumps(losses))


class ParquetRowIterator:
    def __init__(self, file_path, batch_size=1):
        self.parquet_file = pq.ParquetFile(file_path)
        self.batch_iter = self.parquet_file.iter_batches(batch_size=batch_size)
        self.current_batch = None
        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        # If we ran out of batch or it's the first call
        if self.current_batch is None or self.current_idx >= len(self.current_batch):
            try:
                self.current_batch = next(self.batch_iter).to_pandas()
                self.current_idx = 0
            except StopIteration:
                raise StopIteration

        row = self.current_batch.iloc[self.current_idx]
        self.current_idx += 1
        return row


class LazyCluster:
    
    register = []
    edu_embeddings = []
    sizes = []
    cluster_index = 0
    total_size = 0
    ran = False
    
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('all-MiniLM-L12-v2', device="cuda" if cuda.is_available() else "cpu") 
    
    def __init__(
        self,
        dialog : list[str]
    ):
        LazyCluster.register += dialog
        self.i = LazyCluster.cluster_index
        LazyCluster.cluster_index += 1
        LazyCluster.sizes.append((LazyCluster.total_size, LazyCluster.total_size + len(dialog)))
        LazyCluster.total_size += len(dialog)
        
    def __str__(self):
        
        if LazyCluster.ran == False:
            LazyCluster.edu_embeddings = LazyCluster.encoder.encode(LazyCluster.register, batch_size=32, convert_to_numpy=True)
            LazyCluster.ran = True
            
        edu_embeddings = LazyCluster.edu_embeddings[LazyCluster.sizes[self.i][0]:LazyCluster.sizes[self.i][1]]

        # print(self.i, LazyCluster.cluster_index, edu_embeddings.shape)
        if len(edu_embeddings.shape) <= 1:
            return str([])
    
        index = faiss.IndexFlatL2(edu_embeddings.shape[1])
        index.add(edu_embeddings)
        _, I = index.search(edu_embeddings, 2)
        
        if self.i == LazyCluster.cluster_index - 1:
            # print("activated")
            LazyCluster.register = []
            LazyCluster.edu_embeddings = []
            LazyCluster.cluster_index = 0
            LazyCluster.ran = False
            LazyCluster.total_size = 0
            LazyCluster.sizes = []

        return str(json.dumps(I.tolist()))
    
    def __repr__(self):
        
        return self.__str__()

class UniversalParser:
    
    # def __init__(self):
    #     from sentence_transformers import SentenceTransformer
    #     self.encoder = SentenceTransformer('all-MiniLM-L12-v2', device="cuda") 
        
    def __count_elements_in_json_array(file_path):
        with open(file_path, 'rb') as f:
            parser = ijson.items(f, 'item')
            count = sum(1 for _ in parser)
        return count
    
    def __count_lines(filepath):
        with open(filepath, 'rb') as f:
            return sum(1 for _ in f)
    
    # def clustering(self, dialog : list[str]) -> list[list[int]]:
        
    #     enc_time = time()
    #     edu_embeddings = self.encoder.encode(dialog, batch_size=32, convert_to_numpy=True)
    #     print(f"Encoding time: [{time() - enc_time}]")
    #     faiss_time = time()
    #     if len(edu_embeddings.shape) <= 1:
    #         return []
    #     index = faiss.IndexFlatL2(edu_embeddings.shape[1])
    #     index.add(edu_embeddings)
    #     _, I = index.search(edu_embeddings, 2)
    #     print(f"Faiss time: [{time() - faiss_time}]")
    #     return I.tolist()
    
    def flatten(list):
        return [a for v in list for a in v]

    def split_in_edus(dialog, speakers):

        edus = [[a.strip() for a in re.split(r"[.,?!]", d) if len(a)] for d in dialog]
        speakers = UniversalParser.flatten([[speakers[i]] * len(edus[i]) for i in range(len(dialog))])
        edus = UniversalParser.flatten(edus)

        return edus, speakers
    
    def __count_parquet_lines(self, dataset_path : str):
        
        parquet_file = pq.ParquetFile(dataset_path)
        return parquet_file.metadata.num_rows
    
    def __count_csv_lines(self, dataset_path : str):
        with open(dataset_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            return sum(1 for _ in reader)
    
    def process_size(self, dataset_path : str, format : str):
        
        format_to_method = {
            "json": self.__count_elements_in_json_array,
            "line-by-line" : self.__count_lines,
            "parquet": self.__count_parquet_lines,
            "csv" : self.__count_csv_lines
        }
        
        return format_to_method[format](dataset_path)
    
    def convert(
        self,
        dataset_path : str,
        output_path : str,
        start_unit : int = 0,
        to_process : int = None,
        read_format : str = "line-by-line"
    ):
        fieldnames = ["Dialog (EDUs)","Speakers","Positive Pairs","QA","ΔState","Losses"]
        
        i = 0
        avg = 0
        to_process = to_process if to_process != None else self.process_size(dataset_path, read_format)
        
        print("To_process:", to_process)
        
        with open(output_path, "w+") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if start_unit == 0:
                writer.writeheader()

            # Select open mode depending on format
            format_to_reader = {
                "line-by-line": lambda d: open(d, "r"),
                "json" : lambda d: open(d, "r"),
                "parquet" : lambda d: ParquetRowIterator(d),
                "csv" : lambda d: csv.reader(open(d, "r"))
            }
            
            dataset = format_to_reader[read_format](dataset_path)
            if read_format == "csv":
                next(dataset)

            format_to_iterator = {
                "line-by-line" : lambda d: d,
                "json" : lambda d: ijson.items(d, "item"),
                "parquet" : lambda d: d,
                "csv" : lambda d : d
            }
            
            iterator = format_to_iterator[read_format](dataset)
            batch_size = 100
            rows = []
            for unit in iterator:
                i += 1
                if i <= start_unit:
                    continue
                before = time()
                row = self.parse_unit(unit)
                print(f"i: {i}")
                rows.append(row)
                if i % batch_size == 0:
                    writer.writerows(rows)
                    rows = []
                taken = time() - before
                avg = avg * ((i - 1 - start_unit) / (i - start_unit)) + taken * (1/(i - start_unit))
                print(f"Estimated time left: [{datetime.timedelta(seconds=avg * (to_process - i))}]")
                    
    def parse_unit(self, unit : str | dict) -> OutputRow:
        """_summary_

        Args:
            unit (str | dict): the line / json entry to parse

        Returns:
            OutputRow: The row that will be written by the parser
        """
        raise RuntimeError("Not Implemented")


class MediaSumParser(UniversalParser):    

    def parse_unit(self, unit : str):
        
        unit = json.loads(unit)
        
        dialog, speakers = unit["utt"], [s.split("(")[0].split(",")[0].strip() for s in unit["speaker"]]
        dialog, speakers = UniversalParser.split_in_edus(dialog, speakers)
        
        return OutputRow(
            dialog          = dialog,
            speakers        = speakers,
            positive_pairs  = LazyCluster(dialog),
            QA              = None,
            Δ_state         = None
        )


class SODA(UniversalParser):
    
    def parse_unit(self, unit : pd.Series):
        
        edus, speakers = UniversalParser.split_in_edus(unit["dialogue"], unit["speakers"])
        
        return OutputRow(
            dialog          = edus,
            speakers        = speakers,
            positive_pairs  = LazyCluster(edus),
            QA              = None,
            Δ_state         = None
        )
        
class SamSUM(UniversalParser):
    
    def parse_unit(self, unit : list[str]):

        raw_dialog = [a for a in unit[1].split("\n") if ":" in a]
        # print(raw_dialog)
        speakers = [d.split(":")[0] for d in raw_dialog]
        dialog = [d.split(":")[1].strip() for d in raw_dialog]
        
        edus, speakers = UniversalParser.split_in_edus(dialog, speakers)
        
        return OutputRow(
            dialog          = edus,
            speakers        = speakers,
            positive_pairs  = LazyCluster(edus),
            QA              = None,
            Δ_state         = None
        )
