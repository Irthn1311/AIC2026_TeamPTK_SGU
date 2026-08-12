import json
import csv
from pathlib import Path

# Create query objects list
queries = []
groundtruth_all = {}
gt_kis_rows = [["query_id", "video_id", "frame_idx"]]
gt_qa_rows = [["query_id", "video_id", "frame_idx", "answer"]]
gt_trake_rows = [["query_id", "video_id", "event_id", "frame_idx"]]

# ============================================================
# 1. KIS QUERIES (KIS-01 to KIS-14)
# ============================================================
kis_data = [
    {
        "id": "q001_kis_01",
        "text": "Tìm cảnh cháy rừng lớn vào ban đêm, lửa lan dọc sườn đồi và khói dày phủ bầu trời.",
        "video": "L21_V001",
        "pts": 25.0,
        "frame": 750,
    },
    {
        "id": "q002_kis_02",
        "text": "Tìm cảnh một người đàn ông ngồi xổm bên một cột nước trắng phun mạnh thẳng lên từ mặt đất ở khu vực nông thôn.",
        "video": "L21_V001",
        "pts": 227.0,
        "frame": 6810,
    },
    {
        "id": "q003_kis_03",
        "text": "Tìm cảnh một người đàn ông tóc bạc, đeo kính, phát biểu tại bục với cờ Mỹ và biểu tượng ngân hàng trung ương ở phía sau.",
        "video": "L21_V001",
        "pts": 782.0,
        "frame": 23460,
    },
    {
        "id": "q004_kis_04",
        "text": "Tìm cảnh một con hẻm đông người và xe máy, hai bên treo nhiều cờ Việt Nam.",
        "video": "L21_V002",
        "pts": 21.0,
        "frame": 630,
    },
    {
        "id": "q005_kis_05",
        "text": "Tìm cảnh một tháp cổ bằng gạch đã xuống cấp, cây xanh mọc trên phần thân công trình.",
        "video": "L21_V002",
        "pts": 317.0,
        "frame": 9510,
    },
    {
        "id": "q006_kis_06",
        "text": "Tìm cảnh một khu BMX/skatepark ngoài trời vào ban đêm với nhiều dốc trượt và hình vẽ xe đạp trên tường.",
        "video": "L21_V002",
        "pts": 993.0,
        "frame": 29790,
    },
    {
        "id": "q007_kis_07",
        "text": "Tìm đồ họa nền xanh liệt kê số cửa xả đang mở của các hồ thủy điện Hòa Bình, Sơn La và Tuyên Quang.",
        "video": "L21_V003",
        "pts": 71.0,
        "frame": 1775,
    },
    {
        "id": "q008_kis_08",
        "text": "Tìm cận cảnh một thau/chậu tròn chứa rất nhiều cá nhỏ màu bạc.",
        "video": "L21_V003",
        "pts": 359.0,
        "frame": 8975,
    },
    {
        "id": "q009_kis_09",
        "text": "Tìm cảnh một chiếc xe mui trần cổ màu đỏ chở nhiều người đi qua đám đông.",
        "video": "L21_V003",
        "pts": 1030.0,
        "frame": 25750,
    },
    {
        "id": "q010_kis_10",
        "text": "Tìm cảnh một phụ nữ mặc áo xanh họa tiết được phỏng vấn, phía sau có các gói sản phẩm và giấy chứng nhận.",
        "video": "L21_V005",
        "pts": 94.0,
        "frame": 2820,
    },
    {
        "id": "q011_kis_11",
        "text": "Tìm cảnh một người đàn ông đội mũ rơm được phỏng vấn cạnh bờ biển nhiều đá.",
        "video": "L21_V005",
        "pts": 245.0,
        "frame": 7350,
    },
    {
        "id": "q012_kis_12",
        "text": "Tìm cảnh núi lửa đang phun, tạo cột khói rất lớn trên nền trời xanh.",
        "video": "L21_V005",
        "pts": 660.0,
        "frame": 19800,
    },
    {
        "id": "q013_kis_13",
        "text": "Tìm cảnh nữ người dẫn chương trình mặc áo màu be/hồng nhạt đứng một mình trong trường quay.",
        "video": "L21_V006",
        "pts": 103.0,
        "frame": 3090,
    },
    {
        "id": "q014_kis_14",
        "text": "Tìm cảnh một khu chợ/không gian ẩm thực đông người, nhiều quầy chế biến món ăn dưới mái che lớn.",
        "video": "L21_V006",
        "pts": 269.0,
        "frame": 8070,
    },
]

for item in kis_data:
    qid = item["id"]
    vid = item["video"]
    prefix = vid.split("_")[0]
    queries.append({
        "query_id": qid,
        "type": "textual_kis",
        "target_prefix": prefix,
        "text": item["text"]
    })
    groundtruth_all[qid] = {
        "type": "textual_kis",
        "video_id": vid,
        "frame_idx": item["frame"],
        "pts_time": item["pts"]
    }
    gt_kis_rows.append([qid, vid, str(item["frame"])])

# ============================================================
# 2. QA QUERIES (QA-01 to QA-11)
# ============================================================
qa_data = [
    {
        "id": "q015_qa_01",
        "desc": "Cảnh có tấm biển cảnh báo màu vàng-đỏ",
        "question": "Biển đang cảnh báo nguy hiểm gì?",
        "video": "L21_V001",
        "frame": 2250,
        "answer": "Sạt lở nguy hiểm"
    },
    {
        "id": "q016_qa_02",
        "desc": "Cảnh người đàn ông mặc áo sọc đang làm việc với cán bộ",
        "question": "Anh ta đang ngồi phía sau vật gì?",
        "video": "L21_V001",
        "frame": 15870,
        "answer": "Song sắt / khung cửa sắt"
    },
    {
        "id": "q017_qa_03",
        "desc": "Cảnh có người dắt chó trên đường",
        "question": "Có bao nhiêu con chó nhìn thấy rõ?",
        "video": "L21_V001",
        "frame": 34050,
        "answer": "2"
    },
    {
        "id": "q018_qa_04",
        "desc": "Cảnh nữ MC trong trường quay đứng cạnh màn hình lớn",
        "question": "Màn hình lớn cạnh nữ MC đang hiển thị cảnh gì?",
        "video": "L21_V001",
        "frame": 26490,
        "answer": "Một vụ cháy rừng/đám cháy lớn ban đêm"
    },
    {
        "id": "q019_qa_05",
        "desc": "Cảnh sự kiện trên sân khấu",
        "question": "Người đàn ông đứng gần giữa mặc áo khoác màu gì?",
        "video": "L21_V002",
        "frame": 3150,
        "answer": "Trắng"
    },
    {
        "id": "q020_qa_06",
        "desc": "Cảnh nam MC trong trường quay tin tức",
        "question": "Nam MC trong trường quay có đeo kính không?",
        "video": "L21_V002",
        "frame": 17100,
        "answer": "Có"
    },
    {
        "id": "q021_qa_07",
        "desc": "Cảnh thiết bị nông nghiệp nhìn từ trên cao đang chạy",
        "question": "Thiết bị nông nghiệp nhìn từ trên cao đang chạy trên loại khu vực nào?",
        "video": "L21_V002",
        "frame": 26010,
        "answer": "Cánh đồng/ruộng trồng theo luống"
    },
    {
        "id": "q022_qa_08",
        "desc": "Cảnh bữa tiệc sinh nhật trong bản tin",
        "question": "Bữa tiệc sinh nhật trong bản tin được tổ chức cho con vật nào?",
        "video": "L21_V003",
        "frame": 575,
        "answer": "Hà mã"
    },
    {
        "id": "q023_qa_09",
        "desc": "Cảnh phương tiện cỡ lớn ở phía trái khung hình trên đường phố",
        "question": "Phương tiện cỡ lớn ở phía trái khung hình là loại xe gì?",
        "video": "L21_V003",
        "frame": 13775,
        "answer": "Xe tải"
    },
    {
        "id": "q024_qa_10",
        "desc": "Cảnh người đang ở trên cao phía trên thành phố",
        "question": "Người đang ở trên cao phía trên thành phố ngồi trên vật gì?",
        "video": "L21_V003",
        "frame": 28125,
        "answer": "Xích đu"
    },
    {
        "id": "q025_qa_11",
        "desc": "Cảnh cây trồng đang được máy thu hoạch trên đồng",
        "question": "Cây trồng đang được máy thu hoạch trong cảnh là gì?",
        "video": "L21_V005",
        "frame": 540,
        "answer": "Lúa"
    },
]

for item in qa_data:
    qid = item["id"]
    vid = item["video"]
    prefix = vid.split("_")[0]
    queries.append({
        "query_id": qid,
        "type": "qa",
        "target_prefix": prefix,
        "description": item["desc"],
        "question": item["question"]
    })
    groundtruth_all[qid] = {
        "type": "qa",
        "video_id": vid,
        "frame_idx": item["frame"],
        "answer": item["answer"]
    }
    gt_qa_rows.append([qid, vid, str(item["frame"]), item["answer"]])

# ============================================================
# 3. TRAKE QUERIES (TR-01 to TR-10)
# ============================================================
trake_data = [
    {
        "id": "q026_trake_01",
        "activity": "Bản tin thời sự tổng hợp: cháy rừng, cảnh báo và cứu hộ",
        "sport_category": "news",
        "video": "L21_V001",
        "events": [
            {"id": 1, "name": "Cháy rừng ban đêm", "description": "Cảnh cháy rừng lớn vào ban đêm lửa lan dọc sườn đồi khói phủ bầu trời", "hint": "Lửa lan sườn đồi khói dày ban đêm", "frame": 750},
            {"id": 2, "name": "Biển cảnh báo sạt lở", "description": "Tấm biển cảnh báo màu vàng đỏ nguy hiểm sạt lở", "hint": "Biển cảnh báo màu vàng đỏ sạt lở", "frame": 2250},
            {"id": 3, "name": "Cột nước phun mạnh", "description": "Người đàn ông ngồi xổm bên cột nước trắng phun mạnh từ mặt đất", "hint": "Cột nước phun mạnh thẳng lên từ mặt đất", "frame": 6810}
        ]
    },
    {
        "id": "q027_trake_02",
        "activity": "Bản tin thời sự tổng hợp: phỏng vấn, phát biểu và thời sự",
        "sport_category": "news",
        "video": "L21_V001",
        "events": [
            {"id": 1, "name": "Người đàn ông áo sọc sau song sắt", "description": "Người đàn ông mặc áo sọc làm việc với cán bộ ngồi phía sau song sắt", "hint": "Người mặc áo sọc sau khung cửa song sắt", "frame": 15870},
            {"id": 2, "name": "Phát biểu trước cờ Mỹ", "description": "Người đàn ông tóc bạc đeo kính phát biểu tại bục với cờ Mỹ phía sau", "hint": "Tóc bạc đeo kính bục cờ Mỹ", "frame": 23460},
            {"id": 3, "name": "Nữ MC màn hình cháy rừng", "description": "Màn hình lớn cạnh nữ MC hiển thị đám cháy rừng ban đêm", "hint": "Màn hình trường quay đám cháy rừng", "frame": 26490}
        ]
    },
    {
        "id": "q028_trake_03",
        "activity": "Bản tin thời sự tổng hợp: từ trường quay đến đời sống ban đêm",
        "sport_category": "news",
        "video": "L21_V001",
        "events": [
            {"id": 1, "name": "Nữ MC đứng cạnh màn hình cháy rừng", "description": "Nữ MC trong trường quay đứng cạnh màn hình lớn hiển thị vụ cháy rừng", "hint": "Nữ MC trường quay màn hình cháy rừng", "frame": 26490},
            {"id": 2, "name": "Hai con chó được dắt đi", "description": "Cảnh có người dắt hai con chó nhìn thấy rõ trên đường", "hint": "Dắt hai con chó đi dạo", "frame": 34050},
            {"id": 3, "name": "Xe cứu trợ chữ thập đỏ ban đêm", "description": "Chiếc xe cứu trợ hoặc xe chữ thập đỏ di chuyển vào ban đêm", "hint": "Xe chữ thập đỏ ban đêm", "frame": 37080}
        ]
    },
    {
        "id": "q029_trake_04",
        "activity": "Khám phá đời sống và di tích lịch sử",
        "sport_category": "culture",
        "video": "L21_V002",
        "events": [
            {"id": 1, "name": "Hẻm treo nhiều cờ Việt Nam", "description": "Con hẻm đông người và xe máy hai bên treo nhiều cờ Việt Nam", "hint": "Hẻm nhỏ treo cờ đỏ sao vàng", "frame": 630},
            {"id": 2, "name": "Sự kiện sân khấu áo trắng", "description": "Sự kiện trên sân khấu người đàn ông mặc áo khoác màu trắng", "hint": "Sân khấu người đàn ông áo khoác trắng", "frame": 3150},
            {"id": 3, "name": "Tháp cổ bằng gạch", "description": "Tháp cổ bằng gạch xuống cấp cây xanh mọc trên thân", "hint": "Tháp cổ bằng gạch mọc cây", "frame": 9510}
        ]
    },
    {
        "id": "q030_trake_05",
        "activity": "Nhật ký truyền hình: di tích, trường quay và nông nghiệp",
        "sport_category": "culture",
        "video": "L21_V002",
        "events": [
            {"id": 1, "name": "Tháp cổ bằng gạch", "description": "Tháp cổ bằng gạch xuống cấp có cây xanh mọc trên thân", "hint": "Tháp cổ bằng gạch xuống cấp", "frame": 9510},
            {"id": 2, "name": "Nam MC đeo kính", "description": "Nam MC trong trường quay tin tức đeo kính", "hint": "Nam MC đeo kính studio", "frame": 17100},
            {"id": 3, "name": "Máy nông nghiệp nhìn từ trên cao", "description": "Thiết bị nông nghiệp nhìn từ trên cao đang chạy trên đồng", "hint": "Máy nông nghiệp nhìn từ trên cao", "frame": 26010}
        ]
    },
    {
        "id": "q031_trake_06",
        "activity": "Chương trình nhịp sống trẻ và thể thao thể nghiệm",
        "sport_category": "sports",
        "video": "L21_V002",
        "events": [
            {"id": 1, "name": "Nam MC đeo kính", "description": "Nam MC trong trường quay có đeo kính phát biểu", "hint": "Nam MC đeo kính", "frame": 17100},
            {"id": 2, "name": "Máy nông nghiệp trên ruộng", "description": "Thiết bị nông nghiệp chạy trên ruộng trồng theo luống", "hint": "Máy nông nghiệp ruộng theo luống", "frame": 26010},
            {"id": 3, "name": "Khu BMX skatepark ban đêm", "description": "Khu BMX skatepark ngoài trời ban đêm dốc trượt và hình vẽ xe đạp trên tường", "hint": "Skatepark ban đêm hình vẽ xe đạp", "frame": 29790}
        ]
    },
    {
        "id": "q032_trake_07",
        "activity": "Bản tin tổng hợp môi trường và thiên nhiên",
        "sport_category": "nature",
        "video": "L21_V003",
        "events": [
            {"id": 1, "name": "Tiệc sinh nhật hà mã", "description": "Bữa tiệc sinh nhật trong bản tin tổ chức cho con hà mã", "hint": "Bữa tiệc sinh nhật hà mã", "frame": 575},
            {"id": 2, "name": "Đồ họa xả hồ thủy điện", "description": "Đồ họa nền xanh liệt kê số cửa xả đang mở của các hồ thủy điện", "hint": "Đồ họa nền xanh hồ thủy điện", "frame": 1775},
            {"id": 3, "name": "Chậu cá nhỏ màu bạc", "description": "Cận cảnh thau chậu tròn chứa rất nhiều cá nhỏ màu bạc", "hint": "Chậu tròn nhiều cá nhỏ màu bạc", "frame": 8975}
        ]
    },
    {
        "id": "q033_trake_08",
        "activity": "Hành trình nhịp sống thành phố",
        "sport_category": "city",
        "video": "L21_V003",
        "events": [
            {"id": 1, "name": "Thau chậu đầy cá nhỏ", "description": "Thau chậu tròn chứa rất nhiều cá nhỏ màu bạc", "hint": "Chậu tròn chứa cá nhỏ màu bạc", "frame": 8975},
            {"id": 2, "name": "Xe tải lớn trên đường phố", "description": "Phương tiện cỡ lớn xe tải ở phía trái khung hình", "hint": "Xe tải phía trái đường", "frame": 13775},
            {"id": 3, "name": "Xe mui trần cổ màu đỏ", "description": "Chiếc xe mui trần cổ màu đỏ chở nhiều người đi qua đám đông", "hint": "Xe mui trần cổ màu đỏ chở người", "frame": 25750}
        ]
    },
    {
        "id": "q034_trake_09",
        "activity": "Khu trò chơi và giao thông đô thị",
        "sport_category": "city",
        "video": "L21_V003",
        "events": [
            {"id": 1, "name": "Xe tải lớn trên đường", "description": "Phương tiện cỡ lớn xe tải ở phía trái khung hình", "hint": "Xe tải cỡ lớn đường phố", "frame": 13775},
            {"id": 2, "name": "Xe mui trần cổ màu đỏ", "description": "Xe mui trần cổ màu đỏ chở nhiều người qua đám đông", "hint": "Xe mui trần cổ màu đỏ", "frame": 25750},
            {"id": 3, "name": "Người ngồi xích đu trên cao", "description": "Người đang ở trên cao phía trên thành phố ngồi trên xích đu", "hint": "Người ngồi xích đu trên cao thành phố", "frame": 28125}
        ]
    },
    {
        "id": "q035_trake_10",
        "activity": "Phóng sự nông nghiệp và cuộc sống ven biển",
        "sport_category": "agri",
        "video": "L21_V005",
        "events": [
            {"id": 1, "name": "Máy thu hoạch lúa", "description": "Cây trồng lúa đang được máy thu hoạch trên đồng", "hint": "Máy thu hoạch lúa trên đồng", "frame": 540},
            {"id": 2, "name": "Phụ nữ áo xanh họa tiết", "description": "Phụ nữ mặc áo xanh họa tiết được phỏng vấn phía sau có gói sản phẩm", "hint": "Phụ nữ áo xanh phỏng vấn sản phẩm", "frame": 2820},
            {"id": 3, "name": "Người đàn ông đội mũ rơm", "description": "Người đàn ông đội mũ rơm được phỏng vấn cạnh bờ biển nhiều đá", "hint": "Người đàn ông đội mũ rơm bờ biển", "frame": 7350}
        ]
    },
]

for item in trake_data:
    qid = item["id"]
    vid = item["video"]
    prefix = vid.split("_")[0]
    ev_list = []
    gt_events_dict = {}
    for ev in item["events"]:
        ev_list.append({
            "id": ev["id"],
            "name": ev["name"],
            "description": ev["description"],
            "hint": ev["hint"]
        })
        gt_events_dict[str(ev["id"])] = ev["frame"]
        gt_trake_rows.append([qid, vid, str(ev["id"]), str(ev["frame"])])

    queries.append({
        "query_id": qid,
        "type": "trake",
        "target_prefix": prefix,
        "activity": item["activity"],
        "sport_category": item["sport_category"],
        "events": ev_list
    })
    groundtruth_all[qid] = {
        "type": "trake",
        "video_id": vid,
        "events": gt_events_dict
    }

# Save files
queries_dir = Path("datasets/queries")
queries_dir.mkdir(parents=True, exist_ok=True)

with open(queries_dir / "sample_queries.json", "w", encoding="utf-8") as f:
    json.dump(queries, f, ensure_ascii=False, indent=2)

with open(queries_dir / "aic25_b1_queries.json", "w", encoding="utf-8") as f:
    json.dump(queries, f, ensure_ascii=False, indent=2)

gt_dir = Path("datasets/groundtruth")
gt_dir.mkdir(parents=True, exist_ok=True)

with open(gt_dir / "groundtruth_all.json", "w", encoding="utf-8") as f:
    json.dump(groundtruth_all, f, ensure_ascii=False, indent=2)

with open(gt_dir / "groundtruth_kis.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(gt_kis_rows)

with open(gt_dir / "groundtruth_qa.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(gt_qa_rows)

with open(gt_dir / "groundtruth_trake.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(gt_trake_rows)

print(f"Successfully generated {len(queries)} queries ({len(kis_data)} KIS, {len(qa_data)} QA, {len(trake_data)} TRAKE).")
