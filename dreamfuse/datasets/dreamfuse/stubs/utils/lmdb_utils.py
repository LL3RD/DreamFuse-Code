import lmdb
from PIL import Image
import io


class ImageLmdbReader(object):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.txn = None

    def init_txn(self, path):
        env = lmdb.open(
            path,
            max_readers=4,
            readonly=True,
            lock=False,
            map_size=50 * 1024 ** 3,
            readahead=True,
            meminit=False,
        )
        return env.begin(write=False)

    def __call__(self, key):
        if self.txn is None:
            self.txn = self.init_txn(self.lmdb_path)
        data_bytes = self.txn.get(key.encode())
        image = self.parse_image_bytes(data_bytes)
        return image

    def parse_image_bytes(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
                image = image.convert("RGBA")
                white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
                white.paste(image, mask=image.split()[3])
                image = white
            else:
                image = image.convert("RGB")
            return image
        except:
            return image