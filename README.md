# tracking_people_via_DeepSORT

## Трекер людей с отображением пройденного пути.
### 1) deep_sort_tracking_NEW.py 
В качестве детектора используется YOLOv8, натренированная на датасете COCO. Загружается напрямую с сайта Ultralytics. Можно использовать любую версию от Nano до XLarge. Используется Nano, если не указан аргумент.
В качестве  Re-ID embedder используется mobilenet (если не выбрана другая сеть из списка). Возможные сети для использования: mobilenet, torchreid, clip_RN50, clip_RN101, clip_RN50x4", clip_RN50x16, clip_ViT-B/32, clip_ViT-B/16.

Запускать можно непосредственно сам py файл (при этом не получится передать аргумент/флаги в него) либо через терминал.
usage: deep_sort_tracking_NEW.py [-h] [--input INPUT] [--imgsz IMGSZ]
                                 [--model {yolov8n,yolov8s,yolov8m,yolov8l,yolov8x}] [--threshold THRESHOLD]
                                 [--embedder {mobilenet,torchreid,clip_RN50,clip_RN101,clip_RN50x4,clip_RN50x16,clip_ViT-B/32,clip_ViT-B/16}]
                                 [--show]

### 2) deep_sort_tracking.py 
Взят за основу из источника <https://github.com/spmallick/learnopencv/tree/master/Real_Time_Deep_SORT_using_Torchvision_Detectors>
Можно использовать и его. Синтаксис запроса аналогичный. Однако из детекторов используются Region based CNN, поэтому время работы увеличивается. 

## Список использованных ресурсов
1) https://github.com/spmallick/learnopencv/tree/master/Real_Time_Deep_SORT_using_Torchvision_Detectors (https://www.youtube.com/watch?v=GkZRKaQZ_ys)
2) https://www.youtube.com/watch?v=jIRRuGN0j5E
3) https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking/tree/main (https://www.youtube.com/watch?v=9jRRZ-WL698)
4) https://github.com/RizwanMunawar/yolov8-object-tracking
