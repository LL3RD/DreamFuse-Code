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


if __name__ == "__main__":
    lmdb_path = "/mnt/bn/hjj-humanseg-lq/SubjectDriven/Datasets/DreamFuse_Data/DreamFuse80K"
    reader = ImageLmdbReader(lmdb_path)
    json_data = "/mnt/bn/hjj-humanseg-lq/SubjectDriven/Datasets/DreamFuse_Data/DreamFuse80K.json"
    import json
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor

    def process_image(reader, key, img_info_key, data, is_mask=False):
        try:
            image = reader(data[key]["img_mask_info" if is_mask else "img_info"][img_info_key])
        except Exception as e:
            print(f"Error reading {'mask ' if is_mask else ''}image {data[key]['img_mask_info' if is_mask else 'img_info'][img_info_key]}: {e}")

    with open(json_data, "r") as f:
        data = json.load(f)

    with ThreadPoolExecutor(max_workers=32) as executor:
        for key, value in tqdm(data.items(), desc="Processing images"):
            for img_info_key in value["img_info"]:
                executor.submit(process_image, reader, key, img_info_key, data, is_mask=False)
            for img_mask_info_key in value["img_mask_info"]:
                executor.submit(process_image, reader, key, img_mask_info_key, data, is_mask=True)