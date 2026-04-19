# Thesis-Work code fix plan for GNN + teacher pipeline

## Mục tiêu của file này
File này tổng hợp các lỗi/chỗ yếu chính trong pipeline hiện tại và chỉ ra:
1. **File nào cần sửa**
2. **Vấn đề hiện tại là gì**
3. **Vì sao nó làm kết quả train bị méo / khó đọc**
4. **Nên sửa theo hướng nào**
5. **Thứ tự ưu tiên sửa**

---

# 1) Kết luận tổng quan

## 1.1. Solver baseline
- Baseline phía Gurobi **không có dấu hiệu hỏng hoàn toàn**.
- Phần đáng lo hơn nằm ở **teacher dataset -> graph dataset -> GNN training/evaluation**.

## 1.2. Vấn đề chính hiện tại
Hiện tại pipeline GNN có 4 điểm yếu lớn:

### (A) Graph grouping quá thô
- Nhiều teacher rows bị gom thành quá ít graph samples.
- Hậu quả: train/valid/test sample cực ít, metrics không đáng tin.

### (B) Auto-resume checkpoint
- Training đang dễ tự động resume model cũ.
- Nhưng history, best_valid, epoch, optimizer state lại không resume sạch.
- Hậu quả: history khó hiểu, khó biết model thật sự học gì ở run mới.

### (C) Metric đánh giá chưa khớp objective
- Bạn đang train chủ yếu theo `pairwise_rank`.
- Nhưng lại đọc thêm F1 theo threshold 0.5 kiểu classification.
- Hậu quả: có thể xuất hiện tình trạng top1/MRR đẹp nhưng F1 rất xấu.

### (D) Teacher label còn khá nhị phân
- Label hiện tại thiên về `selected_in_rmp`.
- Điều này hợp lệ, nhưng còn nghèo thông tin cho ranking learning.

---

# 2) File cần sửa và lý do

## 2.1. `GNN/build_teacher_graph_dataset.py`

### Vấn đề hiện tại
Hàm `group_key(...)` đang group teacher rows chỉ theo:
- `source_instance`
- `episode`

Điều này quá thô.

Nếu một run có:
- 1 source_instance
- 5 episode

thì dù teacher CSV có hàng trăm rows, sau cùng bạn vẫn chỉ có khoảng 5 graph groups.

### Hậu quả
- `teacher rows` có thể nhiều
- nhưng `n_groups` vẫn rất thấp
- train sample cực ít
- valid sample cực ít
- test sample có thể bằng 0

Đây là nguyên nhân rất mạnh giải thích hiện tượng kiểu:
- 672 rows
- chỉ còn 5 graph groups

### Nên sửa như thế nào
Bạn cần đổi cách group để mỗi graph sample phản ánh đúng **một decision state có ý nghĩa cho GNN**.

## Gợi ý group key tốt hơn
Ưu tiên:
- `source_instance`
- `episode`
- `product`
- `period`

Ví dụ:
```python
def group_key(row):
    source_instance = str(row.get("source_instance") or row.get("instance_id") or "default")
    episode = str(row.get("episode") or row.get("episode_id") or "0")
    product = str(row.get("product") or row.get("sku") or "unknown_product")
    period = str(row.get("period") or row.get("time_period") or "unknown_period")
    return source_instance, episode, product, period
```

### Lợi ích
- Một episode có thể tách thành nhiều graph samples
- số group tăng mạnh
- dataset hữu ích hơn cho train/valid/test

### Lưu ý
Chỉ thêm `product`, `period` nếu teacher CSV thật sự có các field đó và chúng phản ánh đúng một pricing/RMP state.
Nếu sau này bạn có field tốt hơn như:
- `pricing_problem_id`
- `episode_product_period_id`
- `column_pool_state_id`

thì nên dùng các field này thay vì đoán từ `product` và `period`.

---

## 2.2. `irp_gurobi_converted.py`

### Vấn đề hiện tại
Trong `run_teacher_graph_and_gnn_training(...)`, code đang:
- build graph dataset
- rồi nếu `DEFAULT_GNN_CHECKPOINT` tồn tại thì **tự động thêm** `--resume-checkpoint`

Điều này làm train run mới dễ bị nhiễm checkpoint cũ mà người chạy không để ý.

### Hậu quả
- Bạn nghĩ đang train mới
- nhưng thật ra đang fine-tune model cũ
- trong khi history hiện tại không phản ánh sạch quá trình đó

### Nên sửa như thế nào
Thêm cờ điều khiển rõ ràng, ví dụ:

```python
def run_teacher_graph_and_gnn_training(
    teacher_csv_path,
    build_graphs=True,
    train_gnn=True,
    train_epochs=10,
    resume_checkpoint=False,
):
    ...
    if train_gnn:
        cmd = [...]
        if resume_checkpoint:
            checkpoint = _project_path(DEFAULT_GNN_CHECKPOINT)
            if checkpoint.exists():
                cmd.extend(["--resume-checkpoint", str(checkpoint)])
```

### Khuyến nghị
- Mặc định: `resume_checkpoint=False`
- Chỉ bật khi bạn thật sự muốn fine-tune

### Ngoài ra
Nên cho phép truyền custom checkpoint path thay vì luôn bám vào `DEFAULT_GNN_CHECKPOINT`.

---

## 2.3. `GNN/03_train_bigat.py`

### Vấn đề hiện tại số 1: resume không sạch
Hiện tại nếu resume:
- model weights được load
- nhưng `best_valid` lại reset về `inf`
- `history = []`
- epoch chạy lại từ 1
- optimizer state không thấy được resume như một training continuation thật sự

### Hậu quả
- file `training_history.json` chỉ phản ánh session hiện tại
- best model của run mới không được so với best cũ
- khó phân tích learning trajectory thật

### Hướng sửa A: nếu muốn “train mới”
Không truyền `--resume-checkpoint`.

### Hướng sửa B: nếu muốn “resume thật sự”
Bạn cần sửa để:
1. load optimizer state (nếu checkpoint có)
2. load `best_valid`
3. load `history` cũ
4. `start_epoch = last_epoch + 1`

Ví dụ logic:

```python
history = []
start_epoch = 1
best_valid = float("inf")

if args.resume_checkpoint:
    checkpoint = torch.load(...)
    model.load_state_dict(...)
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    best_valid = checkpoint.get("best_valid_loss", float("inf"))
    start_epoch = int(checkpoint.get("last_epoch", 0)) + 1

    history_path = out_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
```

và vòng train:
```python
for epoch in range(start_epoch, start_epoch + args.epochs):
    ...
```

### Vấn đề hiện tại số 2: objective và metric đang lệch nhau
Bạn train với:
- `pairwise_rank`

Nhưng trong `run_epoch(...)` lại luôn log:
- `binary_metrics`
- `topk_accuracy`
- `mrr`
- `mean_positive_rank`

Điều này không sai hoàn toàn, nhưng dễ gây hiểu nhầm nếu bạn đọc F1 như metric chính.

### Nên sửa như thế nào
Khi `objective == "pairwise_rank"`:
- metric chính nên là:
  - `valid_loss`
  - `mrr`
  - `mean_positive_rank`
  - `top1_hit`, `top3_hit`, `top5_hit`
- metric phụ mới là:
  - `precision`
  - `recall`
  - `f1`

### Gợi ý hiển thị
Trong log, tách rõ:
- **Ranking metrics**
- **Binary threshold metrics**

Ví dụ:
```python
if objective == "pairwise_rank":
    primary_metric = "mrr"
else:
    primary_metric = "f1"
```

và trong console/logfile ghi rõ:
- `ranking_valid_mrr`
- `ranking_mean_positive_rank`
- `binary_valid_f1`

để tránh đọc nhầm.

---

## 2.4. `GNN/utilities.py`

### Vấn đề hiện tại số 1: binary_metrics dùng threshold 0.5
Hàm `binary_metrics(...)` đang:
- `probs = sigmoid(scores)`
- `preds = probs >= 0.5`

Đây là logic classification bình thường, nhưng không phải thước đo tốt nhất cho ranking objective.

### Hậu quả
Có thể xảy ra:
- top1 đúng
- mrr tốt
- nhưng F1 thấp

Điều này không nhất thiết nghĩa là model học tệ.

### Nên sửa như thế nào
Giữ `binary_metrics(...)` nhưng xem nó là **secondary metrics**.

Không dùng F1 làm kết luận chính cho pairwise ranking run.

---

### Vấn đề hiện tại số 2: teacher label còn hơi “cứng”
Khi build sample từ teacher rows:
- nếu có `teacher_label` thì dùng
- nếu không thì fallback thành:
  - `1.0` nếu `selected_in_rmp`
  - `0.0` nếu không

Điều này tạo label khá nhị phân.

### Hậu quả
- ranking signal nghèo
- khó phân biệt:
  - cột thực sự tốt
  - cột gần tốt nhưng không được chọn
  - cột rất xấu

### Nên sửa như thế nào
Nếu teacher CSV đã có `teacher_score`, hãy dùng nó mạnh hơn.

## Gợi ý
### Phương án 1: Giữ pairwise ranking nhưng lấy positive/negative từ score mềm hơn
Ví dụ:
- positive = top-ranked or selected
- hard negative = reduced cost tệ / score thấp
- semi-hard negative = gần selected nhưng không selected

### Phương án 2: thử `score_regression`
Nếu `teacher_score` có ý nghĩa và ổn định, bạn có thể thử một nhánh training phụ với:
- objective = `score_regression`

Mục tiêu:
- so sánh `pairwise_rank` vs `score_regression`
- xem objective nào ổn hơn với teacher score hiện có

### Phương án 3: export thêm teacher supervision giàu thông tin hơn
Từ solver side, export thêm:
- reduced cost
- normalized reduced cost
- rank trong candidate pool
- selected_in_rmp
- selected_after_branching
- marginal contribution / improvement proxy

---

# 3) Thứ tự ưu tiên sửa

## Mức ưu tiên 1
### Sửa grouping key
Đây là việc quan trọng nhất.

Nếu không sửa, bạn sẽ vẫn bị:
- teacher rows nhiều
- graph samples quá ít
- GNN không có đủ data để học

---

## Mức ưu tiên 2
### Tắt auto-resume mặc định
Để mỗi run train đều “sạch” và dễ đọc.

---

## Mức ưu tiên 3
### Làm sạch training history / resume workflow
Nếu resume thì resume thật sự.
Nếu train mới thì train mới thật sự.

---

## Mức ưu tiên 4
### Tách ranking metrics và binary metrics
Để khỏi đọc nhầm F1.

---

## Mức ưu tiên 5
### Làm giàu teacher supervision
Cái này quan trọng nhưng có thể làm sau khi 4 bước trên đã sạch.

---

# 4) Patch đề xuất ngắn gọn theo từng file

## File: `GNN/build_teacher_graph_dataset.py`
### Việc cần làm
- sửa `group_key(...)`
- group chi tiết hơn theo decision state

### Mục tiêu
- tăng `n_groups`
- tăng số train/valid/test samples thật

---

## File: `irp_gurobi_converted.py`
### Việc cần làm
- bỏ auto-resume mặc định
- thêm cờ `resume_checkpoint=False`

### Mục tiêu
- kiểm soát run mới vs fine-tune

---

## File: `GNN/03_train_bigat.py`
### Việc cần làm
- nếu resume:
  - load optimizer state
  - load best_valid
  - load history cũ
  - start_epoch tiếp nối
- nếu không resume:
  - clear history đúng nghĩa

### Mục tiêu
- training history sạch
- so sánh run được

---

## File: `GNN/utilities.py`
### Việc cần làm
- giữ `binary_metrics(...)` nhưng coi là metric phụ
- ưu tiên ranking metrics khi objective là `pairwise_rank`
- cân nhắc tận dụng `teacher_score` tốt hơn

### Mục tiêu
- đánh giá đúng thứ model đang học

---

# 5) Sau khi sửa code, nên chạy lại thế nào

## Giai đoạn 1: kiểm tra pipeline sạch
Chạy 1 instance vừa phải:
- 7 stores
- 3 SKU
- 2 đến 3 tháng
- vehicle_capacity khoảng 500

Mục tiêu:
- teacher CSV không rỗng
- `n_groups` tăng rõ rệt
- train/valid/test đều có sample

## Giai đoạn 2: train mới hoàn toàn
- không resume checkpoint
- train vài epoch ngắn
- đọc:
  - valid loss
  - MRR
  - mean positive rank
  - top-k hit

## Giai đoạn 3: mới tính chuyện tăng data lớn hơn
Chỉ khi pipeline sạch thì mới:
- tăng stores
- tăng SKU
- tăng horizon
- gom nhiều runs

---

# 6) Dấu hiệu cho thấy code đã sửa đúng

## Dấu hiệu tốt
- `dataset_summary.json` cho thấy `n_groups` tăng đáng kể
- train/valid/test đều có nhiều sample hơn
- history mỗi run rõ ràng, không lẫn run cũ
- MRR / mean positive rank cải thiện dần
- F1 có thể không hoàn hảo, nhưng không còn là metric gây nhiễu chính

## Dấu hiệu chưa ổn
- rows rất nhiều nhưng groups vẫn rất thấp
- training history vẫn “mất”
- mỗi lần train lại đều không rõ là run mới hay resume
- metric vẫn mâu thuẫn mà không giải thích được

---

# 7) Khuyến nghị thực tế cho bạn lúc này

Đừng sửa tất cả cùng lúc.

## Nên sửa theo thứ tự:
1. `build_teacher_graph_dataset.py`
2. `irp_gurobi_converted.py`
3. `03_train_bigat.py`
4. `utilities.py`

Sau mỗi bước, chạy lại một run nhỏ để kiểm tra.

---

# 8) Chốt

Hiện tại bài của bạn **không phải solver chết hay mô hình GNN sai hoàn toàn**.

Vấn đề lớn nhất là:
- dữ liệu teacher sau grouping bị co lại quá mạnh
- training workflow không sạch vì auto-resume
- metric đang bị đọc lệch mục tiêu học

Sửa đúng 4 điểm trên thì pipeline sẽ dễ tin cậy hơn rất nhiều.

