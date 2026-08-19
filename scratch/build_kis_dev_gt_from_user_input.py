#!/usr/bin/env python3
"""Build immutable kis_dev_gt.json from user-authorized DEV table input."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

RAW_TEXT = """ID\tTruy vấn\tVideo GT\tThời gian GT\tFrame GT đề xuất\tNhánh test\tMức
KIS-01\tTìm cảnh cháy rừng lớn vào ban đêm, lửa lan dọc sườn đồi và khói dày phủ bầu trời.\tL21_V001\t00:25\tf=750 | [720-780]\tVisual scene\tDễ
KIS-02\tTìm cảnh một người đàn ông ngồi xổm bên một cột nước trắng phun mạnh thẳng lên từ mặt đất ở khu vực nông thôn.\tL21_V001\t03:47\tf=6,810 | [6,780-6,840]\tVisual + Object\tTrung bình
KIS-03\tTìm cảnh một người đàn ông tóc bạc, đeo kính, phát biểu tại bục với cờ Mỹ và biểu tượng ngân hàng trung ương ở phía sau.\tL21_V001\t13:02\tf=23,460 | [23,430-23,490]\tVisual + OCR\tTrung bình
KIS-04\tTìm cảnh một con hẻm đông người và xe máy, hai bên treo nhiều cờ Việt Nam.\tL21_V002\t00:21\tf=630 | [600-660]\tVisual scene\tDễ
KIS-05\tTìm cảnh một tháp cổ bằng gạch đã xuống cấp, cây xanh mọc trên phần thân công trình.\tL21_V002\t05:17\tf=9,510 | [9,480-9,540]\tVisual scene\tDễ
KIS-06\tTìm cảnh một khu BMX/skatepark ngoài trời vào ban đêm với nhiều dốc trượt và hình vẽ xe đạp trên tường.\tL21_V002\t16:33\tf=29,790 | [29,760-29,820]\tVisual + Object\tTrung bình
KIS-07\tTìm đồ họa nền xanh liệt kê số cửa xả đang mở của các hồ thủy điện Hòa Bình, Sơn La và Tuyên Quang.\tL21_V003\t01:11\tf=1,775 | [1,750-1,800]\tOCR\tDễ
KIS-08\tTìm cận cảnh một thau/chậu tròn chứa rất nhiều cá nhỏ màu bạc.\tL21_V003\t05:59\tf=8,975 | [8,950-9,000]\tObject\tDễ
KIS-09\tTìm cảnh một chiếc xe mui trần cổ màu đỏ chở nhiều người đi qua đám đông.\tL21_V003\t17:10\tf=25,750 | [25,725-25,775]\tVisual + Object\tDễ
KIS-10\tTìm cảnh một phụ nữ mặc áo xanh họa tiết được phỏng vấn, phía sau có các gói sản phẩm và giấy chứng nhận.\tL21_V005\t01:34\tf=2,820 | [2,790-2,850]\tVisual + OCR\tTrung bình
KIS-11\tTìm cảnh một người đàn ông đội mũ rơm được phỏng vấn cạnh bờ biển nhiều đá.\tL21_V005\t04:05\tf=7,350 | [7,320-7,380]\tVisual scene\tDễ
KIS-12\tTìm cảnh núi lửa đang phun, tạo cột khói rất lớn trên nền trời xanh.\tL21_V005\t11:00\tf=19,800 | [19,770-19,830]\tVisual scene\tDễ
KIS-13\tTìm cảnh nữ người dẫn chương trình mặc áo màu be/hồng nhạt đứng một mình trong trường quay.\tL21_V006\t01:43\tf=3,090 | [3,060-3,120]\tVisual\tDễ
KIS-14\tTìm cảnh một khu chợ/không gian ẩm thực đông người, nhiều quầy chế biến món ăn dưới mái che lớn.\tL21_V006\t04:29\tf=8,070 | [8,040-8,100]\tVisual scene\tTrung bình
KIS-15\tTìm cảnh một hồ bơi ngoài trời màu xanh ngọc với rất nhiều người đang tắm.\tL21_V006\t16:54\tf=30,420 | [30,390-30,450]\tVisual + Object\tDễ
KIS-16\tTìm cảnh đoàn cán bộ đứng kiểm tra khu vực bờ bị sạt lở, phía sau có các khối bê tông chắn sóng.\tL21_V007\t00:16\tf=480 | [450-510]\tVisual scene\tTrung bình
KIS-17\tTìm cảnh nhiều người đội mũ bảo hộ đang làm việc hoặc tham quan trong một nhà xưởng/công trình lớn.\tL21_V007\t04:46\tf=8,580 | [8,550-8,610]\tVisual + Object\tTrung bình
KIS-18\tTìm cảnh một người đàn ông đứng trong vườn chuối và cầm/kiểm tra một buồng hoặc hoa chuối.\tL21_V007\t10:22\tf=18,660 | [18,630-18,690]\tVisual + Object\tDễ
KIS-19\tTìm cảnh nhiều con bò trong chuồng, một số con nằm trên nền đất.\tL21_V008\t00:26\tf=650 | [625-675]\tObject\tDễ
KIS-20\tTìm cảnh hai người dẫn chương trình đứng cạnh nhau trong trường quay, nam mặc áo xanh và nữ mặc áo be.\tL21_V008\t04:56\tf=7,400 | [7,375-7,425]\tVisual\tDễ
KIS-21\tTìm cận cảnh một thiết bị tròn màu đen trong bối cảnh tối, bên phải có nhiều tia lửa bắn ra.\tL21_V008\t16:38\tf=24,950 | [24,925-24,975]\tVisual + Object\tKhó
KIS-22\tTìm cận cảnh nhiều túi lưới chứa các quả trứng màu nhạt xếp chồng lên nhau.\tL21_V009\t00:23\tf=575 | [550-600]\tObject\tDễ
KIS-23\tTìm cảnh một đàn vịt rất đông đang đứng và di chuyển trên cánh đồng ngập nước.\tL21_V009\t05:47\tf=8,675 | [8,650-8,700]\tObject\tDễ
KIS-24\tTìm cảnh một người đeo mặt nạ có khuôn mặt giống Donald Trump đứng giữa đám đông.\tL21_V009\t16:37\tf=24,925 | [24,900-24,950]\tVisual\tKhó
KIS-25\tTìm cảnh một người lớn tuổi ngồi làm thủ tục tại quầy, giữa hai bên có tấm chắn trong suốt.\tL21_V010\t00:22\tf=550 | [525-575]\tVisual scene\tTrung bình
KIS-26\tTìm cảnh một công nhân mặc áo cam đang sơn/trát một bức tường trắng ngoài trời.\tL21_V010\t04:54\tf=7,350 | [7,325-7,375]\tVisual + Action\tDễ
KIS-27\tTìm cảnh nữ MC mặc váy vàng đứng trong trường quay cạnh màn hình đang chiếu một đám đông ban đêm.\tL21_V010\t13:58\tf=20,950 | [20,925-20,975]\tVisual\tTrung bình
KIS-28\tTìm cảnh bác sĩ đeo khẩu trang, mặc áo blouse trắng khám cho một em bé được người đàn ông bế.\tL21_V011\t00:20\tf=500 | [475-525]\tVisual + Object\tDễ
KIS-29\tTìm cảnh một người mặc áo đỏ thao tác trước dãy tủ gửi đồ tự động màu xám có chữ “Giảm CO2”.\tL21_V011\t04:21\tf=6,525 | [6,500-6,550]\tOCR + Object\tTrung bình
KIS-30\tTìm cảnh hai phụ nữ chạy thi trên bãi cỏ trong một cuộc đua vượt chướng ngại vật.\tL21_V011\t15:44\tf=23,600 | [23,575-23,625]\tVisual + Action\tDễ
KIS-31\tTìm cảnh một sườn đồi bị sạt lở, đất đá trượt xuống tạo thành mảng nâu lớn.\tL21_V012\t00:18\tf=540 | [510-570]\tVisual scene\tDễ
KIS-32\tTìm cảnh một phòng học trống với nhiều dãy bàn ghế ngay ngắn.\tL21_V012\t03:25\tf=6,150 | [6,120-6,180]\tVisual scene\tDễ
KIS-33\tTìm cảnh dòng thác/ghềnh nước đục chảy rất mạnh, phía dưới có đông khách tham quan.\tL21_V012\t13:59\tf=25,170 | [25,140-25,200]\tVisual scene\tTrung bình
KIS-34\tTìm cảnh thiết bị/buồng nâng màu đỏ làm việc trên cao giữa tán cây để kiểm tra cây xanh.\tL21_V013\t00:22\tf=660 | [630-690]\tVisual + Object\tTrung bình
KIS-35\tTìm cảnh hai MC trong trường quay, nữ mặc trang phục đen-trắng và nam mặc sơ mi tối màu.\tL21_V013\t04:52\tf=8,760 | [8,730-8,790]\tVisual\tDễ
KIS-36\tTìm cảnh đám đông đang ăn mừng và giơ một khung chứng nhận Guinness World Records.\tL21_V013\t15:24\tf=27,720 | [27,690-27,750]\tOCR + Visual\tTrung bình
KIS-37\tTìm cảnh một sàn/buồng nâng màu đỏ đang hoạt động trên cao trong tán cây xanh.\tL21_V014\t00:22\tf=660 | [630-690]\tVisual + Object\tTrung bình
KIS-38\tTìm cảnh quay từ trên cao một thị trấn ven sông/biển với rất nhiều tàu thuyền đậu dọc bờ.\tL21_V014\t04:50\tf=8,700 | [8,670-8,730]\tVisual scene\tDễ
KIS-39\tTìm cảnh quay từ trên cao một nhóm cá voi/cá lớn đang bơi trên mặt biển.\tL21_V014\t16:45\tf=30,150 | [30,120-30,180]\tVisual + Object\tTrung bình
KIS-40\tTìm cảnh quay từ trên cao một khu du lịch xanh mướt với hồ nước màu xanh ngọc giữa địa hình khô cằn ở Oman.\tL21_V015\t00:26\tf=780 | [750-810]\tVisual scene\tTrung bình
KIS-41\tTìm cận cảnh một rổ nhựa đỏ chứa đầy cá nhỏ màu bạc.\tL21_V015\t06:31\tf=11,730 | [11,700-11,760]\tObject\tDễ
KIS-42\tTìm cảnh một tay đua xe đạp đứng trên bục màu cam của Zwift và giơ cúp lên cao.\tL21_V015\t20:25\tf=36,750 | [36,720-36,780]\tOCR + Visual\tTrung bình
KIS-43\tTìm cảnh một tuyến đường trung tâm thành phố có nhiều cây xanh và xe máy đang lưu thông.\tL21_V016\t00:21\tf=630 | [600-660]\tVisual scene\tDễ
KIS-44\tTìm cảnh trường quay có một nam MC ở bàn dẫn và một nữ MC đứng ở phía bên phải.\tL21_V016\t05:16\tf=9,480 | [9,450-9,510]\tVisual\tDễ
KIS-45\tTìm cảnh một quảng trường châu Âu có thảm hoa khổng lồ nhiều màu trải trên mặt đất.\tL21_V016\t15:48\tf=28,440 | [28,410-28,470]\tVisual scene\tTrung bình
KIS-46\tTìm cảnh nhiều xe tải/xe thương mại đang ở bên trong một trung tâm đăng kiểm.\tL21_V017\t00:19\tf=475 | [450-500]\tObject + Scene\tDễ
KIS-47\tTìm cảnh quay từ trên cao một nút giao thông lớn với rất nhiều ô tô và xe máy di chuyển theo nhiều hướng.\tL21_V017\t02:13\tf=3,325 | [3,300-3,350]\tVisual scene\tTrung bình
KIS-48\tTìm cảnh nhiều khán giả đứng dọc hai bên một tuyến đường có cọc tiêu màu cam để xem sự kiện/đua xe.\tL21_V017\t14:15\tf=21,375 | [21,350-21,400]\tVisual scene\tTrung bình
KIS-49\tTìm cảnh một xe tải/xe cứu trợ màu trắng có biểu tượng chữ thập đỏ chạy hoặc dừng trước dãy cửa hàng vào ban đêm.\tL21_V001\t20:36\tf=37,080 | [37,050-37,110]\tVisual + Object\tTrung bình
KIS-50\tTìm cảnh một vận động viên trượt ván đang đứng trên ván tại một skatepark ngoài trời lớn.\tL21_V015\t21:17\tf=38,310 | [38,280-38,340]\tVisual + Action\tDễ"""

HOLDOUT_VIDEOS = {"L21_V006", "L21_V013", "L21_V014", "L21_V002"}
DEV_VIDEOS = {
    "L21_V005", "L21_V010", "L21_V012", "L21_V016",
    "L21_V009", "L21_V007", "L21_V017", "L21_V003",
    "L21_V011", "L21_V015", "L21_V008", "L21_V001"
}


def build_kis_dev_gt() -> None:
    lines = [l.strip() for l in RAW_TEXT.strip().splitlines() if l.strip()]
    records = []
    holdout_records = []

    for line in lines[1:]:
        parts = line.split("\t")
        qid, query_vi, vid, ref_time, raw_frame_str, branch, diff = parts

        center_match = re.search(r"f=([\d,]+)", raw_frame_str)
        interval_match = re.search(r"\[([\d,]+)-([\d,]+)\]", raw_frame_str)
        assert center_match and interval_match, f"Malformed frame string: {raw_frame_str}"

        center_f = int(center_match.group(1).replace(",", ""))
        start_f = int(interval_match.group(1).replace(",", ""))
        end_f = int(interval_match.group(2).replace(",", ""))
        assert start_f <= center_f <= end_f, f"Center frame {center_f} not in [{start_f}..{end_f}]"

        if vid in HOLDOUT_VIDEOS:
            holdout_records.append((qid, vid))
            continue

        assert vid in DEV_VIDEOS, f"Unknown video {vid}"
        records.append({
            "query_id": qid,
            "query_vi": query_vi,
            "video_id": vid,
            "reference_timestamp": ref_time,
            "proposed_frame_center": center_f,
            "start_frame": start_f,
            "end_frame": end_f,
            "branch": branch,
            "difficulty": diff,
            "split": "DEV",
        })

    print(f"Total raw lines parsed : {len(lines) - 1}")
    print(f"Holdout excluded ({len(holdout_records)}): {[q for q, v in holdout_records]}")
    print(f"DEV queries extracted  : {len(records)}")

    # Strict Assertions
    assert len(records) == 38, f"Expected exactly 38 DEV queries, got {len(records)}"
    assert len(set(r["query_id"] for r in records)) == 38, "Duplicate DEV query IDs!"
    for r in records:
        assert r["split"] == "DEV", f"Invalid split {r['split']}"
        assert r["start_frame"] <= r["end_frame"], f"Invalid interval in {r}"
        assert r["video_id"] in DEV_VIDEOS, f"Non-DEV video {r['video_id']}"

    out_obj = {
        "schema_version": "kis_dev_gt_v1",
        "benchmark_id": "system_tai-l21-150-diagnostic-v1",
        "split": "DEV",
        "query_count": 38,
        "queries": records,
    }

    out_path = Path("systems/system_tai/benchmarks/l21_150_diagnostic/kis_dev_gt.json")
    out_bytes = (json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    out_path.write_bytes(out_bytes)
    out_sha = hashlib.sha256(out_bytes).hexdigest()

    print(f"Successfully wrote immutable DEV-only GT artifact: {out_path}")
    print(f"File size: {len(out_bytes)} bytes")
    print(f"Artifact SHA256: {out_sha}")


if __name__ == "__main__":
    build_kis_dev_gt()
