import struct
import os
import zstandard as zstd
import array
import random
from typing import Callable, Iterable
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class StringDataFile:
    """
    int32_t => compress level\n
    uint32_t => dict_size\n
    *dict_size* => dict data\n\n

    chunks...\n
    
    chunk format:\n
    4 bytes => Little-endian unsigned int : size of current chunk (exclude this 4 bytes)\n
    *size* bytes => bytes data compressed by compress_data function\n
    """
    def __init__(self , file_path: str) -> None:
        self.file_path: str = file_path
        # open files
        self.data_file = open(os.path.join(self.file_path , "datas") , mode="ab+")
        self.offsets_file = open(os.path.join(self.file_path , "offsets") , mode="ab+")

        self.offsets: array.array[int] = array.array("Q")   # int64 offsets

        self.data_file.seek(0, os.SEEK_END)     # get data_file size
        self._size = self.data_file.tell()

        self.data_file.seek(0)     # seek to the start
        self.offsets_file.seek(0)
        # read compress level
        _compress_level_bytes = self.data_file.read(4)
        if not _compress_level_bytes:
            raise ValueError("no compress level infomation.")
        self.compress_level: int = struct.unpack("<i" , _compress_level_bytes)[0]

        # read dict
        _dict_size_bytes = self.data_file.read(4)
        if not _dict_size_bytes:
            raise ValueError("no dict size infomation.")
        self.dict_size:int = struct.unpack("<I" , _dict_size_bytes)[0]
        _dict_buffer = self.data_file.read(self.dict_size)
        self.compressor_dict = zstd.ZstdCompressionDict(_dict_buffer)

        self.compressor = zstd.ZstdCompressor(self.compress_level , dict_data=self.compressor_dict)
        self.decompressor = zstd.ZstdDecompressor(dict_data=self.compressor_dict)

        self.offsets: array.array[int] = self.read_offset_file()
    
    def __len__(self) -> int:
        return len(self.offsets)
    
    def __getitem__(self , idx: int) -> str:
        return self._read_chunk(self.offsets[idx])
    
    @property
    def size(self):
        return self._size
    
    def read_offset_file(self):
        _offsets = array.array("Q") # unsigned long long
        _offsets.append(self.dict_size + 8)     # compress_level int + dict_size int + dict_size
        self.offsets_file.seek(0)
        while curr_bytes := self.offsets_file.read(4):
            curr = struct.unpack("<I" , curr_bytes)[0]
            _offsets.append(curr + 4 + _offsets[-1])
        return _offsets[:-1]

    def _read_chunk(self , offset: int) -> str:
        self.data_file.seek(offset)  # seek to given offset
        _size = struct.unpack("<I" , self.data_file.read(4))[0]
        _raw_data = self.data_file.read(_size)
        _data = self.decompress_data(_raw_data)
        return _data
    
    def _write_chunk(self , data: str) -> None:
        self.data_file.seek(0, os.SEEK_END) # seek to end
        self.offsets_file.seek(0, os.SEEK_END)
        self.offsets.append(self.data_file.tell())
        _compressed_data = self.compress_data(data)
        _size = len(_compressed_data)
        _size_bytes = struct.pack("<I" , _size)
        self.data_file.write(_size_bytes)           # write size
        self.data_file.write(_compressed_data)      # write data
        self.offsets_file.write(_size_bytes)        # write size in offsets file
        self._size = self.data_file.tell()

    def compress_data(self , data: str) -> bytes:
        return self.compressor.compress(data.encode(encoding="utf-8" , errors="ignore"))
    
    def decompress_data(self , raw_data: bytes) -> str:
        return self.decompressor.decompress(raw_data).decode(encoding="utf-8" , errors="ignore")

    def __del__(self):
        self.data_file.close()
        self.offsets_file.close

    @staticmethod
    def create_file(
            file_path: str,
            compress_level: int,
            dict_size: int,
            dict_sample_rows: int,
            data_iter: Iterable[str] | list[str],
            max_file_size: int,
            filter: Callable[[str], bool] | None = None,
            map: Callable[[str], str] | None = None,
        ) -> None:
        if filter is None:
            filter = lambda _ : True
        if map is None:
            map = lambda s : s

        # prepare dict training data
        dict_samples: list[bytes] = []
        for text in data_iter:
            if filter(text):
                text = map(text)
            else:
                continue
            dict_samples.append(text.encode())
            if len(dict_samples) % 100 == 0:
                print(f"prepare dict samples, {len(dict_samples)}/{dict_sample_rows}", end="\r")
            if len(dict_samples) >= dict_sample_rows:
                break
        print("\ncomplete. start write datas")

        zstd_dict = zstd.train_dictionary(dict_size=dict_size , samples=dict_samples, level=compress_level)     # train dict

        # make file
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        with open(os.path.join(file_path , "datas") , "ab+") as file:
            file.seek(0 , os.SEEK_END)
            if file.tell() != 0:
                raise ValueError("file is not empty")

            # write compress level
            file.write(struct.pack("<i" , compress_level))

            # write dict datas
            file.write(struct.pack("<I" , len(zstd_dict)))
            file.write(zstd_dict.as_bytes())
        
        # writing datas
        data_file = StringDataFile(file_path)
        for text in data_iter:
            if filter(text):
                text = map(text)
            else:
                continue
            data_file._write_chunk(text)
            if len(data_file) % 100 == 0:
                print(f"rows:{len(data_file)},file_size:{data_file.size}", end="\r")
            if data_file.size >= max_file_size:
                break
        print("\ncompleted")


class wordDataset(Dataset):
    def __init__(self, datafile_path: str, tokenizer_path: str, read_len: int, pad_tok: int, sep_tok: int) -> None:
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.datafile_path = datafile_path
        self.pad_tok = pad_tok
        self.sep_tok = sep_tok
        self.read_len = read_len
        self.datafile = None
        self.tokenizer = None

        _datafile = StringDataFile(datafile_path)
        self.length = len(_datafile)

    def __len__(self):
        return self.length
    def __getitem__(self, index) -> torch.Tensor:
        if self.tokenizer is None:
            self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        if self.datafile is None:
            self.datafile = StringDataFile(self.datafile_path)
        
        ids = []
        while len(ids) < self.read_len:
            ids.extend(self.tokenizer.encode(self.datafile[index]).ids + [self.sep_tok])
            index += 1
        return torch.tensor(ids[:self.read_len], dtype=torch.long)

def train_tokenizer(
        datafile_path: str,
        tokenizer_path: str, 
        target_vocabs: int, 
        vocab_step_size: int, 
        vocab_step_samples: int, 
        max_token_len: int | None = None
    ):
    r"""
    a function to train the tokenizer

    :param target_vocabs: the final target of the tokenizer.
    :param vocab_step_size: Determine how many new tokens to be add in a step.
    :param vocab_step_samples: Determine how many samples(StringDataFile rows) will be use to train in a step.
    """
    special_tokens = ["[BOS]" , "[EOS]" , "[PAD]" , "[UNK]" , "[MASK]"]

    datafile = StringDataFile(datafile_path)
    def data_sampler():
        for _ in range(vocab_step_samples):
            yield datafile[random.randint(0 , len(datafile) - 1)]


    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()    # type: ignore
    tokenizer.decoder = decoders.ByteLevel()    # type: ignore

    curr_vocabs = 256 + len(special_tokens)
    while True:
        trainer = trainers.BpeTrainer(
            vocab_size=min(curr_vocabs + vocab_step_size, target_vocabs),
            min_frequency=2,
            special_tokens=special_tokens,
            max_token_length=max_token_len
        )
        tokenizer.train_from_iterator(data_sampler(), trainer, length=vocab_step_samples)
        curr_vocabs += vocab_step_size
        if curr_vocabs >= target_vocabs:
            break

    tokenizer.save(tokenizer_path)