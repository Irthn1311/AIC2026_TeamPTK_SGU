"""
BTC Object Search & Index Module (Branch 3)
===========================================
Builds an object-level search index (Inverted Index / BM25) for BTC detected objects
and enables filtering & scoring keyframes by object queries.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.retrieval.logging_utils import setup_logger

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    tokens = re.findall(r"\w+", text)
    return tokens


SYNONYMS_MAP: dict[str, list[str]] = {
    # -------------------------------------------------------------
    # 1. Watercraft & Marine / Sông nước, Tàu thuyền
    # -------------------------------------------------------------
    "thuyền": ["boat", "watercraft", "canoe", "ship", "tàu", "thuyền máy", "ghe", "xuồng"],
    "boat": ["thuyền", "watercraft", "tàu", "thuyền máy", "canoe", "ship", "xuồng", "ghe"],
    "tàu": ["boat", "ship", "watercraft", "thuyền", "vessel"],
    "thuyền máy": ["boat", "motorboat", "watercraft", "thuyền"],
    "ca nô": ["boat", "canoe", "watercraft", "motorboat"],
    "ghe": ["boat", "watercraft", "thuyền"],
    "xuồng": ["boat", "watercraft", "thuyền", "canoe"],
    "tàu thủy": ["ship", "boat", "watercraft", "vessel"],
    "tàu cá": ["boat", "ship", "watercraft", "fish"],
    "watercraft": ["boat", "thuyền", "tàu", "ship"],
    "ship": ["tàu", "ship", "boat", "watercraft"],
    "sông": ["boat", "watercraft", "ship", "tree", "plant", "water", "river"],
    "biển": ["boat", "watercraft", "ship", "beach", "ocean", "sea", "water"],
    "bờ sông": ["river", "water", "boat", "tree"],
    "bến tàu": ["boat", "ship", "watercraft", "dock", "port"],
    "bến cảng": ["ship", "boat", "watercraft", "port", "dock"],

    # -------------------------------------------------------------
    # 2. Vehicles & Transportation / Phương tiện giao thông
    # -------------------------------------------------------------
    "xe": ["car", "vehicle", "land vehicle", "ô tô", "xe hơi", "xe máy", "motorcycle", "truck", "bus", "van", "suv"],
    "car": ["xe", "ô tô", "xe hơi", "land vehicle", "vehicle", "suv"],
    "ô tô": ["car", "vehicle", "land vehicle", "xe hơi", "xe", "suv"],
    "xe hơi": ["car", "vehicle", "land vehicle", "ô tô", "xe", "suv"],
    "xe con": ["car", "vehicle", "land vehicle", "ô tô", "xe hơi"],
    "xe máy": ["motorcycle", "vehicle", "land vehicle", "xe", "scooter", "motorbike"],
    "xe mô tô": ["motorcycle", "vehicle", "land vehicle", "xe"],
    "xe gắn máy": ["motorcycle", "vehicle", "land vehicle", "xe"],
    "xe tải": ["truck", "vehicle", "land vehicle", "xe", "van"],
    "truck": ["xe tải", "xe", "vehicle", "land vehicle"],
    "xe buýt": ["bus", "vehicle", "land vehicle", "xe"],
    "xe bus": ["bus", "vehicle", "land vehicle", "xe"],
    "bus": ["xe buýt", "xe bus", "vehicle", "land vehicle", "xe"],
    "xe khách": ["bus", "van", "vehicle", "land vehicle", "xe"],
    "xe cứu thương": ["ambulance", "vehicle", "car", "hospital", "doctor", "nurse"],
    "xe cấp cứu": ["ambulance", "vehicle", "car", "hospital", "doctor"],
    "ambulance": ["xe cứu thương", "xe cấp cứu", "vehicle", "car"],
    "xe cứu hỏa": ["fire truck", "truck", "vehicle", "firefighter"],
    "xe chữa cháy": ["fire truck", "truck", "vehicle", "firefighter"],
    "fire truck": ["xe cứu hỏa", "xe chữa cháy", "truck", "vehicle"],
    "xe cảnh sát": ["police car", "car", "police officer", "police", "vehicle"],
    "xe taxi": ["taxi", "cab", "car", "vehicle"],
    "taxi": ["xe taxi", "taxi", "car", "vehicle"],
    "xe đạp": ["bicycle", "bike", "vehicle"],
    "bicycle": ["xe đạp", "bike", "vehicle"],
    "xe van": ["van", "vehicle", "car", "truck"],
    "van": ["xe van", "van", "vehicle", "car"],
    "suv": ["xe", "car", "ô tô", "xe hơi", "vehicle"],
    "máy bay": ["airplane", "aircraft", "vehicle", "plane", "helicopter"],
    "phi cơ": ["airplane", "aircraft", "vehicle", "plane"],
    "trực thăng": ["helicopter", "aircraft", "airplane", "vehicle"],
    "airplane": ["máy bay", "aircraft", "vehicle", "plane"],
    "tàu hỏa": ["train", "railway", "vehicle"],
    "xe lửa": ["train", "railway", "vehicle"],
    "train": ["tàu hỏa", "xe lửa", "railway", "vehicle"],
    "vehicle": ["xe", "car", "land vehicle", "ô tô", "truck", "bus", "motorcycle"],
    "giao thông": ["land vehicle", "vehicle", "car", "motorcycle", "bus", "traffic sign", "traffic light", "traffic"],

    # -------------------------------------------------------------
    # 3. Roads, Traffic & Infrastructure / Giao thông & Hạ tầng
    # -------------------------------------------------------------
    "đường": ["land vehicle", "vehicle", "car", "motorcycle", "tree", "building", "street", "road"],
    "đường phố": ["street", "road", "building", "house", "car", "motorcycle"],
    "phố": ["building", "house", "land vehicle", "vehicle", "car", "street", "road"],
    "vỉa hè": ["building", "house", "person", "tree", "sidewalk"],
    "ngõ": ["building", "house", "door", "window"],
    "hẻm": ["building", "house", "door", "window"],
    "cầu": ["bridge", "river", "road", "water"],
    "cây cầu": ["bridge", "river", "road", "water"],
    "bridge": ["cầu", "cây cầu", "road", "river"],
    "đường hầm": ["tunnel", "road"],
    "ngã tư": ["crossroad", "intersection", "traffic light", "road", "street"],
    "ngã ba": ["intersection", "road", "street"],
    "biển báo": ["traffic sign", "sign", "poster", "billboard"],
    "biển chỉ dẫn": ["traffic sign", "sign", "billboard"],
    "biển giao thông": ["traffic sign", "sign", "traffic light"],
    "traffic sign": ["biển báo", "biển giao thông", "sign"],
    "đèn giao thông": ["traffic light", "traffic sign", "sign"],
    "đèn đỏ": ["traffic light", "sign"],
    "traffic light": ["đèn giao thông", "đèn đỏ", "sign"],
    "rào chắn": ["barrier", "cone", "fence"],
    "cọc tiêu": ["cone", "barrier"],

    # -------------------------------------------------------------
    # 4. People & Occupations / Con người & Nghề nghiệp
    # -------------------------------------------------------------
    "người": ["person", "man", "woman", "human face", "human", "clothing", "đàn ông", "phụ nữ"],
    "person": ["người", "man", "woman", "human face", "human", "clothing"],
    "đàn ông": ["man", "person", "người", "human face", "suit"],
    "man": ["người", "đàn ông", "person", "human face"],
    "phụ nữ": ["woman", "person", "người", "human face", "girl", "dress"],
    "woman": ["người", "phụ nữ", "person", "human face", "girl"],
    "cô gái": ["girl", "woman", "person", "người"],
    "chàng trai": ["boy", "man", "person", "người"],
    "em bé": ["baby", "child", "kid", "person"],
    "trẻ em": ["child", "children", "kid", "boy", "girl", "person"],
    "học sinh": ["student", "person", "man", "woman", "girl", "boy", "clothing", "backpack"],
    "sinh viên": ["student", "person", "man", "woman", "clothing"],
    "người già": ["elderly", "person", "man", "woman"],
    "đông người": ["person", "man", "woman", "human face", "clothing", "crowd", "people"],
    "đám đông": ["person", "man", "woman", "human face", "clothing", "crowd", "people"],
    "crowd": ["đám đông", "đông người", "person", "people"],
    "dân chúng": ["crowd", "people", "person"],
    "người dẫn chương trình": ["presenter", "person", "microphone", "suit", "television", "news"],
    "mc": ["presenter", "person", "microphone", "suit", "television"],
    "presenter": ["người dẫn chương trình", "mc", "person", "microphone"],
    "phóng viên": ["reporter", "journalist", "person", "microphone", "camera", "clothing"],
    "nhà báo": ["reporter", "journalist", "person", "microphone", "camera"],
    "cảnh sát": ["police officer", "police", "person", "officer", "uniform", "hat", "cap"],
    "công an": ["police officer", "police", "person", "officer", "uniform", "hat", "cap"],
    "police officer": ["cảnh sát", "công an", "person", "officer"],
    "bác sĩ": ["doctor", "person", "man", "woman", "clothing", "nurse", "hospital", "clinic"],
    "y tá": ["nurse", "person", "woman", "doctor", "hospital", "clinic"],
    "doctor": ["bác sĩ", "person", "hospital", "nurse"],
    "nurse": ["y tá", "person", "doctor", "hospital"],
    "công nhân": ["worker", "person", "helmet", "uniform"],
    "thợ": ["worker", "person", "helmet"],
    "worker": ["công nhân", "thợ", "person", "helmet"],
    "lính cứu hỏa": ["firefighter", "person", "helmet", "fire truck"],
    "firefighter": ["lính cứu hỏa", "person", "helmet"],
    "người bán hàng": ["vendor", "seller", "person", "shop", "store", "market"],
    "khách hàng": ["customer", "person", "shop", "store"],
    "người đi bộ": ["pedestrian", "person", "sidewalk", "street"],
    "người lái xe": ["driver", "person", "car", "motorcycle", "bus", "truck"],

    # -------------------------------------------------------------
    # 5. Fashion, Apparel & Accessories / Thời trang, Phụ kiện
    # -------------------------------------------------------------
    "mũ": ["hat", "helmet", "cap"],
    "nón": ["hat", "helmet", "cap"],
    "hat": ["mũ", "nón", "cap"],
    "mũ bảo hiểm": ["helmet", "hat", "motorcycle", "bike"],
    "nón bảo hiểm": ["helmet", "hat", "motorcycle"],
    "helmet": ["mũ bảo hiểm", "nón bảo hiểm", "hat", "safety"],
    "mũ lưỡi trai": ["cap", "hat"],
    "kính": ["glasses", "sunglasses", "spectacles"],
    "kính mắt": ["glasses", "sunglasses"],
    "kính râm": ["sunglasses", "glasses"],
    "kính cận": ["glasses", "spectacles"],
    "glasses": ["kính", "kính mắt", "sunglasses"],
    "sunglasses": ["kính râm", "kính mát", "glasses"],
    "cà vạt": ["tie", "necktie", "suit"],
    "caravat": ["tie", "necktie", "suit"],
    "tie": ["cà vạt", "caravat", "necktie", "suit"],
    "đồng hồ": ["watch", "wristwatch", "clock"],
    "đồng hồ đeo tay": ["watch", "wristwatch"],
    "watch": ["đồng hồ", "đồng hồ đeo tay", "wristwatch"],
    "vòng tay": ["bracelet", "jewelry"],
    "lắc tay": ["bracelet", "jewelry"],
    "bracelet": ["vòng tay", "lắc tay"],
    "vòng cổ": ["necklace", "jewelry"],
    "dây chuyền": ["necklace", "jewelry"],
    "necklace": ["vòng cổ", "dây chuyền"],
    "nhẫn": ["ring", "jewelry"],
    "thắt lưng": ["belt"],
    "dây nịt": ["belt"],
    "belt": ["thắt lưng", "dây nịt"],
    "quần áo": ["clothing", "shirt", "dress", "suit", "jacket", "pants"],
    "trang phục": ["clothing", "suit", "dress", "uniform"],
    "clothing": ["quần áo", "shirt", "áo", "quần", "dress", "suit"],
    "áo": ["shirt", "clothing", "t-shirt", "jacket", "coat", "dress", "suit"],
    "áo sơ mi": ["shirt", "clothing", "suit", "tie"],
    "shirt": ["áo sơ mi", "áo", "clothing"],
    "áo thun": ["t-shirt", "shirt", "clothing"],
    "áo phông": ["t-shirt", "shirt", "clothing"],
    "t-shirt": ["áo thun", "áo phông", "shirt", "clothing"],
    "áo khoác": ["jacket", "coat", "clothing"],
    "jacket": ["áo khoác", "coat", "clothing"],
    "áo vest": ["suit", "jacket", "tie", "clothing"],
    "bộ vest": ["suit", "jacket", "tie", "clothing"],
    "âu phục": ["suit", "clothing", "tie"],
    "suit": ["bộ vest", "áo vest", "âu phục", "tie", "clothing"],
    "quần": ["pants", "trousers", "clothing"],
    "pants": ["quần", "trousers", "clothing"],
    "váy": ["dress", "skirt", "clothing", "woman", "girl"],
    "đầm": ["dress", "clothing", "woman", "girl"],
    "chân váy": ["skirt", "dress", "clothing"],
    "dress": ["váy", "đầm", "clothing"],
    "áo dài": ["dress", "clothing", "woman"],
    "túi": ["bag", "handbag", "backpack"],
    "túi xách": ["handbag", "bag", "purse"],
    "handbag": ["túi xách", "túi", "bag"],
    "bag": ["túi", "túi xách", "handbag", "backpack"],
    "ba lô": ["backpack", "bag"],
    "cặp sách": ["backpack", "bag", "briefcase"],
    "backpack": ["ba lô", "cặp sách", "bag"],
    "ví": ["wallet", "purse", "bag"],
    "giày": ["shoes", "footwear", "boots", "sneakers"],
    "dép": ["sandals", "slippers", "shoes"],
    "shoes": ["giày", "dép", "footwear"],
    "găng tay": ["glove", "gloves"],
    "bao tay": ["glove", "gloves"],
    "glove": ["găng tay", "bao tay", "gloves"],
    "gloves": ["găng tay", "bao tay", "glove"],
    "khẩu trang": ["mask", "face mask"],
    "mask": ["khẩu trang", "face mask"],
    "áo phao": ["life jacket", "safety"],
    "life jacket": ["áo phao", "safety"],
    "ô": ["umbrella"],
    "dù": ["umbrella"],
    "umbrella": ["ô", "dù"],

    # -------------------------------------------------------------
    # 6. Meeting, Indoor, Dining & Household Objects / Đồ dùng bàn họp, Ăn uống, Nội thất
    # -------------------------------------------------------------
    "phòng": ["building", "house", "chair", "table", "window", "door", "room", "office"],
    "phòng họp": ["room", "office", "table", "chair", "flower", "bottle", "cup", "meeting"],
    "hội trường": ["hall", "room", "chair", "table", "stage", "screen", "banner"],
    "văn phòng": ["office", "room", "desk", "table", "chair", "computer", "laptop"],
    "bàn": ["table", "desk", "chair"],
    "bàn họp": ["table", "chair", "desk", "flower", "bottle", "cup", "room"],
    "bàn làm việc": ["desk", "table", "chair", "laptop", "computer"],
    "bàn ăn": ["table", "chair", "bowl", "plate", "cup", "bottle"],
    "table": ["bàn", "desk", "chair"],
    "desk": ["bàn làm việc", "bàn", "table"],
    "ghế": ["chair", "seat", "table"],
    "ghế ngồi": ["chair", "seat", "table"],
    "ghế sofa": ["couch", "sofa", "chair"],
    "chair": ["ghế", "seat", "table"],
    "hoa": ["flower", "flowers", "plant", "vase", "bouquet"],
    "bình hoa": ["flower", "vase", "plant", "table", "pot"],
    "lọ hoa": ["flower", "vase", "plant", "table"],
    "bông hoa": ["flower", "flowers", "plant"],
    "chậu hoa": ["flower", "pot", "plant", "vase"],
    "lẵng hoa": ["flower", "flowers", "bouquet", "basket"],
    "flower": ["hoa", "bình hoa", "lọ hoa", "bông hoa", "plant", "vase"],
    "flowers": ["hoa", "bình hoa", "lọ hoa", "plant", "flower"],
    "vase": ["bình hoa", "lọ hoa", "flower", "pot"],
    "chai": ["bottle", "container"],
    "chai nước": ["bottle", "water bottle", "cup", "table"],
    "bình nước": ["bottle", "water bottle", "kettle", "pitcher"],
    "bottle": ["chai", "chai nước", "bình nước", "lọ", "cup"],
    "cốc": ["cup", "glass", "mug", "table", "bottle"],
    "ly": ["cup", "glass", "mug", "table", "bottle"],
    "tách": ["cup", "tea cup", "mug", "table"],
    "tách trà": ["cup", "tea cup", "tea pot", "table"],
    "cup": ["cốc", "ly", "tách", "tách trà", "glass", "mug"],
    "glass": ["ly", "cốc", "glass", "cup"],
    "ấm trà": ["tea pot", "teapot", "kettle", "cup", "table"],
    "bình trà": ["tea pot", "teapot", "kettle", "cup"],
    "tea pot": ["ấm trà", "bình trà", "teapot", "cup", "table"],
    "teapot": ["ấm trà", "bình trà", "tea pot", "cup"],
    "bát": ["bowl", "dish", "plate", "table"],
    "chén": ["bowl", "cup", "dish", "table"],
    "tô": ["bowl", "dish", "table"],
    "bowl": ["bát", "chén", "tô", "dish"],
    "đĩa": ["plate", "dish", "table"],
    "dĩa": ["plate", "fork", "dish", "table"],
    "plate": ["đĩa", "dĩa", "dish", "table"],
    "dish": ["đĩa", "món ăn", "plate", "bowl"],
    "thìa": ["spoon", "table"],
    "muỗng": ["spoon", "table"],
    "đũa": ["chopsticks", "table"],
    "hộp": ["box", "container", "package"],
    "hộp quà": ["gift box", "box", "package", "gift"],
    "quà": ["gift box", "gift", "box"],
    "thùng": ["box", "carton", "container"],
    "thùng carton": ["box", "carton", "container"],
    "box": ["hộp", "thùng", "hộp quà", "gift box", "container"],
    "gift box": ["hộp quà", "quà", "box", "package"],
    "giấy": ["paper", "document", "book"],
    "tài liệu": ["document", "paper", "book", "file"],
    "sổ": ["notebook", "book", "paper"],
    "vở": ["notebook", "book", "paper"],
    "bút": ["pen", "pencil"],
    "viết": ["pen", "pencil"],
    "sách": ["book", "bible", "paper"],
    "book": ["sách", "quyển sách", "bible"],
    "bible": ["kinh thánh", "sách", "book"],
    "tủ": ["cabinet", "wardrobe", "cupboard", "furniture"],
    "kệ": ["shelf", "rack", "furniture"],
    "cửa": ["door", "gate", "building", "house"],
    "cửa sổ": ["window", "building", "house"],
    "door": ["cửa", "cửa ra vào", "building"],
    "window": ["cửa sổ", "building", "house"],
    "rèm": ["curtain", "window"],
    "đèn": ["lamp", "light", "lighting", "ceiling light"],
    "đèn bàn": ["lamp", "table lamp", "light"],
    "lamp": ["đèn", "đèn bàn", "light"],
    "tranh": ["painting", "picture", "art", "poster"],
    "ảnh": ["photo", "picture", "poster", "frame"],

    # -------------------------------------------------------------
    # 7. Electronics, Tech & Media / Điện tử, Thiết bị & Truyền thông
    # -------------------------------------------------------------
    "tivi": ["television", "tv", "screen", "display", "monitor"],
    "ti vi": ["television", "tv", "screen", "display", "monitor"],
    "vô tuyến": ["television", "tv", "screen"],
    "màn hình": ["display", "screen", "television", "tv", "computer", "monitor"],
    "màn hình tv": ["television", "tv", "screen", "display"],
    "màn hình led": ["screen", "display", "billboard", "television"],
    "màn chiếu": ["screen", "projector", "display"],
    "television": ["tivi", "ti vi", "vô tuyến", "tv", "screen", "display", "monitor"],
    "tv": ["tivi", "ti vi", "television", "screen", "monitor"],
    "screen": ["màn hình", "display", "television", "monitor"],
    "display": ["màn hình", "screen", "television", "monitor"],
    "monitor": ["màn hình", "screen", "display", "computer"],
    "máy tính": ["computer", "laptop", "pc", "screen", "monitor", "keyboard"],
    "laptop": ["laptop", "computer", "notebook", "screen"],
    "máy tính xách tay": ["laptop", "computer"],
    "máy tính để bàn": ["computer", "desktop", "monitor", "pc"],
    "computer": ["máy tính", "laptop", "pc", "monitor", "screen"],
    "chuột": ["mouse", "computer"],
    "bàn phím": ["keyboard", "computer", "laptop"],
    "điện thoại": ["mobile phone", "telephone", "phone", "smartphone", "cellphone"],
    "điện thoại di động": ["mobile phone", "phone", "smartphone", "cellphone"],
    "smartphone": ["điện thoại", "phone", "mobile phone"],
    "phone": ["điện thoại", "mobile phone", "telephone", "smartphone"],
    "mobile phone": ["điện thoại", "điện thoại di động", "phone", "smartphone"],
    "telephone": ["điện thoại", "phone"],
    "máy ảnh": ["camera", "photo"],
    "máy quay": ["camcorder", "camera", "video camera"],
    "camera": ["máy ảnh", "máy quay", "camera"],
    "micro": ["microphone", "mic"],
    "mic": ["microphone", "micro"],
    "micrô": ["microphone", "micro", "mic"],
    "microphone": ["micro", "mic", "micrô", "presenter", "reporter"],
    "loa": ["speaker", "audio", "sound"],
    "speaker": ["loa", "audio", "sound"],
    "tai nghe": ["headphones", "earphones", "headset"],
    "máy in": ["printer", "computer"],
    "máy chiếu": ["projector", "screen"],

    # -------------------------------------------------------------
    # 8. Buildings, Real Estate & Places / Công trình, Địa điểm & Thương mại
    # -------------------------------------------------------------
    "nhà": ["house", "building", "skyscraper", "tower", "tòa nhà", "ngôi nhà"],
    "ngôi nhà": ["house", "home", "building"],
    "house": ["nhà", "ngôi nhà", "building", "tòa nhà"],
    "tòa nhà": ["skyscraper", "building", "tower", "house", "nhà"],
    "cao ốc": ["skyscraper", "building", "tower", "tòa nhà"],
    "tòa tháp": ["tower", "skyscraper", "building"],
    "tháp": ["tower", "skyscraper", "building"],
    "skyscraper": ["tòa nhà", "cao ốc", "nhà", "building", "tower"],
    "tower": ["tháp", "tòa tháp", "tòa nhà", "skyscraper", "building"],
    "building": ["tòa nhà", "nhà", "house", "building", "skyscraper", "tower"],
    "chung cư": ["apartment", "building", "house"],
    "cửa hàng": ["building", "house", "window", "door", "billboard", "poster", "store", "shop"],
    "tiệm": ["store", "shop", "building", "cửa hàng"],
    "quán": ["building", "house", "window", "door", "chair", "table", "cafe", "restaurant"],
    "quán ăn": ["restaurant", "cafe", "food", "table", "chair"],
    "quán cà phê": ["cafe", "coffee shop", "table", "chair", "cup"],
    "nhà hàng": ["restaurant", "dining", "table", "chair", "food"],
    "siêu thị": ["building", "house", "window", "door", "billboard", "supermarket", "store", "shop"],
    "shophouse": ["building", "house", "window", "door", "billboard", "tòa nhà", "nhà"],
    "mặt bằng": ["building", "house", "window", "door", "tòa nhà", "nhà"],
    "chợ": ["building", "house", "person", "man", "woman", "clothing", "market"],
    "gian hàng": ["building", "house", "poster", "billboard", "clothing", "booth", "stall"],
    "bách hóa": ["building", "house", "store", "shop", "supermarket"],
    "shop": ["building", "house", "window", "door", "store", "cửa hàng"],
    "store": ["building", "house", "window", "door", "shop", "cửa hàng"],
    "supermarket": ["building", "house", "window", "door", "siêu thị"],
    "bệnh viện": ["hospital", "building", "house", "person", "doctor", "nurse", "ambulance"],
    "phòng khám": ["clinic", "hospital", "building", "person", "doctor", "nurse", "chair", "table"],
    "hospital": ["bệnh viện", "phòng khám", "building", "doctor", "nurse"],
    "clinic": ["phòng khám", "bệnh viện", "building", "doctor"],
    "trường học": ["school", "building", "classroom", "student", "teacher"],
    "lớp học": ["classroom", "school", "room", "desk", "chair", "student"],
    "công viên": ["park", "tree", "plant", "bench", "grass"],
    "bảo tàng": ["museum", "building", "art", "exhibition"],
    "museum": ["bảo tàng", "building", "art"],
    "sân vận động": ["stadium", "field", "sports"],
    "sân bay": ["airport", "airplane", "aircraft", "terminal"],
    "nhà ga": ["station", "train station", "train"],
    "bến xe": ["bus station", "bus", "station"],

    # -------------------------------------------------------------
    # 9. Signs, Media, Symbols & Flags / Biển hiệu, Biểu tượng & Cờ
    # -------------------------------------------------------------
    "biển hiệu": ["sign", "signboard", "billboard", "poster", "banner"],
    "bảng hiệu": ["sign", "signboard", "billboard", "poster", "banner"],
    "biển quảng cáo": ["billboard", "poster", "banner", "sign"],
    "áp phích": ["poster", "billboard", "banner", "sign"],
    "băng rôn": ["banner", "poster", "sign", "billboard"],
    "poster": ["áp phích", "poster", "billboard", "banner", "sign"],
    "banner": ["băng rôn", "banner", "poster", "sign"],
    "billboard": ["biển quảng cáo", "billboard", "sign", "poster"],
    "sign": ["biển báo", "biển hiệu", "bảng hiệu", "sign", "traffic sign"],
    "cờ": ["flag", "symbol", "national flag"],
    "lá cờ": ["flag", "symbol"],
    "cờ đỏ": ["flag", "red flag"],
    "cờ tổ quốc": ["flag", "national flag"],
    "cờ việt nam": ["flag", "national flag"],
    "flag": ["cờ", "lá cờ", "cờ đỏ", "symbol", "national flag"],
    "logo": ["logo", "symbol", "icon", "sign"],
    "biểu tượng": ["symbol", "logo", "icon"],
    "icon": ["biểu tượng", "icon", "logo"],
    "bản đồ": ["map", "chart", "diagram"],

    # -------------------------------------------------------------
    # 10. Nature, Environment, Weather & Animals / Thiên nhiên & Động vật
    # -------------------------------------------------------------
    "cây": ["tree", "plant", "trees", "forest", "cây cối", "cây xanh"],
    "cây xanh": ["tree", "plant", "trees", "cây cối"],
    "cây cối": ["tree", "plant", "forest"],
    "tree": ["cây", "plant", "palm tree", "cây cối", "cây xanh", "forest"],
    "plant": ["cây", "tree", "plant", "flower"],
    "rừng": ["forest", "tree", "trees", "plant", "mountain"],
    "forest": ["rừng", "tree", "trees", "plant"],
    "núi": ["mountain", "hill", "nature"],
    "đồi": ["hill", "mountain", "nature"],
    "mountain": ["núi", "hill", "nature"],
    "bầu trời": ["sky", "cloud", "clouds"],
    "sky": ["bầu trời", "sky", "cloud"],
    "mây": ["cloud", "clouds", "sky"],
    "cloud": ["mây", "cloud", "sky"],
    "clouds": ["mây", "cloud", "sky"],
    "mặt trời": ["sun", "sunset", "sunrise", "sky"],
    "mặt trăng": ["moon", "night", "sky"],
    "nước": ["water", "river", "sea", "ocean", "lake"],
    "water": ["nước", "river", "sea", "ocean"],
    "mưa": ["rain", "water", "storm"],
    "khói": ["smoke", "fire"],
    "smoke": ["khói", "fire"],
    "lửa": ["fire", "flame", "smoke"],
    "fire": ["lửa", "flame", "smoke"],
    "cháy": ["fire", "smoke", "flame"],
    "ngập lụt": ["flood", "water", "rain"],
    "lũ lụt": ["flood", "water", "rain"],
    "flood": ["ngập lụt", "lũ lụt", "water"],
    "cánh đồng": ["field", "farmland", "rice", "paddy", "grass", "pasture", "cow", "yak"],
    "đồng lúa": ["rice", "paddy", "field", "farmland", "plant"],
    "ruộng lúa": ["rice", "paddy", "field", "farmland", "plant"],
    "đồng cỏ": ["pasture", "grass", "field", "cow", "yak", "horse"],
    "nông dân": ["farmer", "farmland", "person", "hat"],
    # Động vật / Animals & Livestock
    "trâu": ["yak", "cow", "cattle", "bull", "ox", "buffalo", "bison", "livestock"],
    "con trâu": ["yak", "cow", "cattle", "bull", "ox", "buffalo", "bison", "livestock"],
    "đàn trâu": ["yak", "cow", "cattle", "bull", "ox", "buffalo", "bison", "livestock", "herd", "pasture", "field"],
    "1 đàn trâu": ["yak", "cow", "cattle", "bull", "ox", "buffalo", "bison", "livestock", "herd"],
    "bò": ["cow", "cattle", "bull", "calf", "ox", "yak", "buffalo", "livestock"],
    "con bò": ["cow", "cattle", "bull", "calf", "ox", "yak", "buffalo", "livestock"],
    "đàn bò": ["cow", "cattle", "bull", "calf", "ox", "yak", "buffalo", "livestock", "herd", "pasture", "field"],
    "1 đàn bò": ["cow", "cattle", "bull", "calf", "ox", "yak", "buffalo", "livestock", "herd"],
    "đàn": ["herd", "flock", "group", "crowd", "livestock", "cow", "yak", "buffalo"],
    "gia súc": ["livestock", "cattle", "cow", "yak", "buffalo", "bull", "pig", "sheep", "goat", "horse"],
    "đàn gia súc": ["livestock", "cattle", "cow", "yak", "buffalo", "herd", "pasture"],
    "yak": ["trâu", "bò", "đàn trâu", "đàn bò", "con trâu", "con bò", "buffalo", "cow", "cattle", "ox", "bull"],
    "cow": ["bò", "trâu", "đàn bò", "đàn trâu", "con bò", "con trâu", "cattle", "yak", "bull", "calf"],
    "buffalo": ["trâu", "con trâu", "đàn trâu", "yak", "cow", "cattle"],
    "cattle": ["gia súc", "bò", "trâu", "đàn bò", "đàn trâu", "cow", "yak", "bull", "ox"],
    "bull": ["bò", "trâu", "cow", "cattle", "yak", "ox"],
    "ox": ["bò", "trâu", "bò cày", "cow", "cattle", "yak", "bull"],
    "livestock": ["gia súc", "đàn gia súc", "trâu", "bò", "heo", "lợn", "cow", "yak", "buffalo", "pig"],
    "chó": ["dog", "puppy", "hound", "bulldog", "pet"],
    "con chó": ["dog", "puppy", "hound", "bulldog", "pet"],
    "đàn chó": ["dog", "puppy", "hound"],
    "dog": ["chó", "con chó", "pet", "bulldog"],
    "mèo": ["cat", "kitten", "feline", "pet"],
    "con mèo": ["cat", "kitten", "feline", "pet"],
    "cat": ["mèo", "con mèo", "pet"],
    "chim": ["bird", "pigeon", "poultry", "sky"],
    "con chim": ["bird", "pigeon", "poultry"],
    "đàn chim": ["bird", "pigeon", "flock"],
    "bồ câu": ["pigeon", "bird"],
    "bird": ["chim", "con chim", "pigeon"],
    "cá": ["fish", "goldfish", "fishing", "seafood"],
    "con cá": ["fish", "goldfish", "fishing"],
    "đàn cá": ["fish", "goldfish", "fishing"],
    "câu cá": ["fishing", "fish"],
    "fish": ["cá", "con cá", "đàn cá", "goldfish", "fishing"],
    "ngựa": ["horse", "stallion", "mare", "pony"],
    "con ngựa": ["horse", "stallion", "mare"],
    "horse": ["ngựa", "con ngựa"],
    "lợn": ["pig", "swine", "hog", "pork"],
    "heo": ["pig", "swine", "hog", "pork"],
    "con lợn": ["pig", "swine", "hog"],
    "con heo": ["pig", "swine", "hog"],
    "pig": ["lợn", "heo", "con lợn", "con heo"],
    "gà": ["chicken", "rooster", "hen", "poultry", "bird"],
    "con gà": ["chicken", "rooster", "hen", "bird"],
    "chicken": ["gà", "con gà", "rooster", "hen"],
    "vịt": ["duck", "poultry", "bird"],
    "con vịt": ["duck", "poultry", "bird"],
    "duck": ["vịt", "con vịt"],
    "dê": ["goat", "ram"],
    "con dê": ["goat", "ram"],
    "cừu": ["sheep", "lamb"],
    "con cừu": ["sheep", "lamb"],
    "voi": ["elephant"],
    "con voi": ["elephant"],
    "hổ": ["tiger"],
    "cọp": ["tiger"],
    "sư tử": ["lion"],
    "khỉ": ["monkey", "ape"],
}


class ObjectIndex:
    """Inverted Index & BM25 Matcher for Object Entities with Bilingual Expansion."""

    def __init__(self, corpus_path: str | Path):
        self.corpus_path = Path(corpus_path)
        self.df = pd.read_parquet(self.corpus_path) if self.corpus_path.suffix.lower() == ".parquet" else pd.read_csv(self.corpus_path)
        self._inverted_index: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._build_index()

    def _build_index(self) -> None:
        logger.info("Building Object Inverted Index from %d records...", len(self.df))
        for row_idx, row in self.df.iterrows():
            search_text = str(row.get("search_text", ""))
            object_scores = row.get("object_scores", {})

            if isinstance(object_scores, str):
                try:
                    object_scores = json.loads(object_scores)
                except Exception:
                    object_scores = {}

            tokens = _tokenize(search_text)
            unique_tokens = set(tokens)

            for token in unique_tokens:
                scores_list = [float(s) for s in object_scores.values() if s is not None]
                max_obj_score = max(scores_list, default=0.5)
                self._inverted_index[token].append((row_idx, max_obj_score))

    def search(self, query: str, top_k: int = 50) -> pd.DataFrame:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return pd.DataFrame()

        # Stopwords to ignore in object matching
        stopwords = {
            "tìm", "cảnh", "các", "và", "ở", "trong", "phóng", "sự", "về", "mặt", "bằng", "theo", "cho", "với", "từ",
            "find", "scenes", "of", "and", "in", "on", "a", "the", "is", "are", "with", "for", "about", "to"
        }
        filtered_query_tokens = [t for t in query_tokens if t not in stopwords and len(t) > 1]
        if not filtered_query_tokens:
            filtered_query_tokens = query_tokens

        # Expand query tokens with bilingual synonyms
        expanded_tokens: set[str] = set(filtered_query_tokens)
        query_lower = query.lower()

        # Check multi-word phrase keys in SYNONYMS_MAP
        for syn_key, syn_targets in SYNONYMS_MAP.items():
            if syn_key in query_lower:
                for target in syn_targets:
                    expanded_tokens.update(_tokenize(target))

        for token in list(filtered_query_tokens):
            if token in SYNONYMS_MAP:
                for syn in SYNONYMS_MAP[token]:
                    expanded_tokens.update(_tokenize(syn))

        scores: dict[int, float] = defaultdict(float)
        match_counts: dict[int, int] = defaultdict(int)

        for token in expanded_tokens:
            if token in self._inverted_index:
                # Direct tokens get full weight 1.0, synonym tokens get weight 0.75
                weight_factor = 1.0 if token in filtered_query_tokens else 0.75
                for row_idx, obj_score in self._inverted_index[token]:
                    scores[row_idx] += (obj_score + 0.5) * weight_factor
                    match_counts[row_idx] += 1

        if not scores:
            return pd.DataFrame()

        # Rank rows by score (weighted confidence score FIRST, match count second)
        ranked_indices = sorted(scores.keys(), key=lambda idx: (scores[idx], match_counts[idx]), reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            row = self.df.iloc[idx].to_dict()
            row["rank"] = rank
            row["object_match_score"] = round(scores[idx], 4)
            row["matched_terms_count"] = match_counts[idx]
            results.append(row)

        return pd.DataFrame(results)


def build_object_index(
    objects_data_path: str | Path,
    output_dir: str | Path,
    logger_inst=None,
) -> tuple[ObjectIndex, dict[str, Any]]:
    objects_data_path = Path(objects_data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_inst = logger_inst or setup_logger("object_index")
    started = time.time()

    index_inst = ObjectIndex(objects_data_path)

    meta = {
        "corpus_path": str(objects_data_path),
        "total_keyframe_records": len(index_inst.df),
        "unique_vocab_size": len(index_inst._inverted_index),
        "elapsed_seconds": round(time.time() - started, 2),
    }

    meta_path = output_dir / "l21_object_index_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log_inst.info("Built Object Index with %d vocab tokens for %d records", meta["unique_vocab_size"], meta["total_keyframe_records"])

    return index_inst, meta
