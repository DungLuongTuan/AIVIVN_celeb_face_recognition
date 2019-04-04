# ARCface + SVM cho bài toán nhận diện người nổi tiếng
Mã nguồn chương trình đạt hạng 4 cuộc thi nhận diện người nổi tiếng do AIVIVN tổ chức

## Requirements
- python==3.5.2
- scikit-learn==0.20.3
- pandas==0.23.1
- scikit-image==0.14.2
- scipy==1.0.0
- mxnet==1.4.0.post0

## Giải pháp
### Xử lý dữ liệu
dữ liệu gồm 1000 nhãn trong đó số lượng ảnh của từng nhãn trong khoảng từ 1-16, trong đó có một vài nhãn còn thuộc cùng 1 người, và có ảnh bị gán nhãn sai trong dữ liệu. Tuy nhiên mình không xử lý gì với 2 vấn đề này mà chỉ thực hiện augment để thêm dữ liệu cho những nhãn có < 3 ảnh, sử dụng xoay, thêm nhiễu và flip ảnh.

### Face Embedding
Face embedding mình sử dụng ở đây là arcface (theo repo: https://github.com/deepinsight/insightface)

### Mô hình
- Mình coi bài toán như một bài phân loại, và sử dụng ngưỡng trên đầu ra để dự đoán người lạ (nhãn 1000)
- Mô hình mình lựa chọn là SVM (sklearn.linearSVC), có tunning để lựa chọn tham số phù hợp với dữ liệu
- Ngưỡng để xác định người lạ được lựa chọn trên giá trị score của decision_function trong sklearn.linearSVC. Mình thực hiện submit một vài lần để dự đoán ra số người lạ, rồi sau đó xác định ngưỡng dựa theo số lượng người lạ mà mình dự đoán (cuối cùng chọn threshold = -0.55). Với những ảnh có score <= ngưỡng, nhãn 1000 được đẩy lên đầu và tiếp đến là 4 nhãn có score cao nhất, với trường hợp score > ngưỡng, nhãn 1000 đưọc đặt tại vị trí 3.

### Sử dụng thêm dữ liệu từ tập test
Mình sử dụng mô hình tốt nhất đã có để dự đoán nhãn trên tập public test, rồi dùng một phần dữ liệu mà mô hình dự đoán với độ chính xác cao để làm dữ liệu thêm (~14k ảnh), việc này gây thêm nhiễu cho mô hình nhưng lại tăng được đáng kể dữ liệu. Sau đó mình train lại mô hình với tập train + dữ liệu thêm này, kết quả thu được tốt hơn so với trước đó.

## Chạy chương trình
Clone insightface
```
git clone https://github.com/deepinsight/insightface.git
```

Sinh embedding
```
python3 prepare_data.py --data_path="thư mục data cuộc thi"
```

```
python3 augment_data
```

```
python3 gen_emb.py
```

Huấn luyện mô hình
```
python3 arcface+linearSVC.py --mode="normal"
```

Thêm dữ liệu
```
python3 add_data.py
```

Huấn luyện lại mô hình với dữ liệu thêm
```
python3 arcface+linearSVC.py --mode="add"
```