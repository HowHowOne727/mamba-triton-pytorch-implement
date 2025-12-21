import struct
import os
import zstandard as zstd
import array


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
        self.file = open(os.path.join(self.file_path , "datas") , mode="ab+")

        self.offsets: array.array[int] = array.array("Q")   # int64 offsets

        self.file.seek(0)     # seek to the start
        # read compress level
        _compress_level_bytes = self.file.read(4)
        if not _compress_level_bytes:
            raise ValueError("no compress level infomation.")
        self.compress_level: int = struct.unpack("<i" , _compress_level_bytes)[0]

        # read dict
        _dict_size_bytes = self.file.read(4)
        if not _dict_size_bytes:
            raise ValueError("file is empty.")
        self.dict_size:int = struct.unpack("<I" , _dict_size_bytes)[0]
        _dict_buffer = self.file.read(self.dict_size)
        self.compressor_dict = zstd.ZstdCompressionDict(_dict_buffer)

        self.compressor = zstd.ZstdCompressor(self.compress_level , dict_data=self.compressor_dict)
        self.decompressor = zstd.ZstdDecompressor(dict_data=self.compressor_dict)

        self.offsets = self.read_offset_file()
    
    def __len__(self) -> int:
        return len(self.offsets)
    
    def __getitem__(self , idx: int) -> str:
        return self._read_chunk(self.offsets[idx])
    
    @property
    def size(self):
        return self._size
    
    def read_offset_file(self):
        _offsets = array.array("Q")
        if os.path.exists(os.path.join(self.file_path , "offsets")):
            _offsets.append(self.dict_size + 8)
            with open(os.path.join(self.file_path , "offsets") , "rb") as file:
                while curr_bytes := file.read(4):
                    curr = struct.unpack("<I" , curr_bytes)[0]
                    _offsets.append(curr + _offsets[-1])
            return _offsets
        
        print("making offsets file...")
        self.file.seek(self.dict_size + 8)
        # init offsets
        while _size_bytes := self.file.read(4):
            _offsets.append(self.file.tell() - 4)   # -4 size bytes
            _size = struct.unpack("<I" , _size_bytes)[0]
            self.file.seek(_size , os.SEEK_CUR)     # seek to next chunk
        self.file.seek(0 , os.SEEK_END)
        self._size:int = self.file.tell()

        with open(os.path.join(self.file_path , "offsets") , "wb") as file:
            for i in range(len(_offsets) - 1):
                file.write(struct.pack("<I" , _offsets[i + 1] - _offsets[i]))
        
        return _offsets

    def _read_chunk(self , offset: int) -> str:
        self.file.seek(offset)  # seek to given offset
        _size = struct.unpack("<I" , self.file.read(4))[0]
        _raw_data = self.file.read(_size)
        _data = self.decompress_data(_raw_data)
        return _data
    
    def _write_chunk(self , data: str) -> None:
        self.file.seek(0 , os.SEEK_END) # seek to end
        self.offsets.append(self.file.tell())
        _compressed_data = self.compress_data(data)
        _size = len(_compressed_data)
        _size_bytes = struct.pack("<I" , _size)
        self.file.write(_size_bytes)         # write size
        self.file.write(_compressed_data)    # write data
        self._size = self.file.tell()

    def compress_data(self , data: str) -> bytes:
        return self.compressor.compress(data.encode(encoding="utf-8" , errors="ignore"))
    
    def decompress_data(self , raw_data: bytes) -> str:
        return self.decompressor.decompress(raw_data).decode(encoding="utf-8" , errors="ignore")

    def __del__(self):
        self.file.close()

    @classmethod
    def create_file(cls , file_path: str, compress_level: int, dict_size: int, dict_samples: list[bytes]):
        zstd_dict = zstd.train_dictionary(dict_size=dict_size , samples=dict_samples, level=compress_level)
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
        return cls(file_path)