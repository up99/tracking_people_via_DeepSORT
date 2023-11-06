# tracking_people_via_DeepSORT

## Задача 1: Трекер людей с отображением пройденного пути.
### 1) deep_sort_tracking_NEW.py (!!!)
В качестве детектора используется YOLOv8, натренированная на датасете COCO. Загружается напрямую с сайта Ultralytics. Можно использовать любую версию от Nano до XLarge. Используется Nano, если не задана модель явно.
В качестве  Re-ID embedder используется mobilenet (если не выбрана другая сеть из списка). Возможные сети для использования: mobilenet, torchreid, clip_RN50, clip_RN101, clip_RN50x4, clip_RN50x16, clip_ViT-B/32, clip_ViT-B/16.

Запускать можно непосредственно сам py файл (при этом не получится передать аргумент/флаги в него) либо через терминал. Очевидно, что лучший результат будет достигнут при использовании большей версии YOLOv8 (например, XLarge, а не Nano). Однако это приведёт к потреблению больших ресурсов -> замедлению инференса.

usage: deep_sort_tracking_NEW.py [-h] [--input INPUT] [--imgsz IMGSZ]
                                 [--model {yolov8n,yolov8s,yolov8m,yolov8l,yolov8x}] [--threshold THRESHOLD]
                                 [--embedder {mobilenet,torchreid,clip_RN50,clip_RN101,clip_RN50x4,clip_RN50x16,clip_ViT-B/32,clip_ViT-B/16}]
                                 [--show]

Пример работы deep_sort_tracking_NEW.py представлен ниже (использованы стандартные параметры: YOLOv8 Nano + mobilenet):

https://github.com/up99/tracking_people_via_DeepSORT/assets/62401614/8f503ace-9737-4236-a47b-41097349b1d5



### 2) deep_sort_tracking.py 
Взят за основу из источника <https://github.com/spmallick/learnopencv/tree/master/Real_Time_Deep_SORT_using_Torchvision_Detectors>

Можно использовать и его. Синтаксис запроса аналогичный. Однако из детекторов используются Region based CNN, поэтому время работы увеличивается. Кроме того, не реализована функция отрисовки трека.

## Задача 2: Отображение пути сверху (bird-eye view)
За основу взят видеоролик: https://www.youtube.com/watch?v=jvgmnJspjoA

## Список использованных ресурсов
1) https://github.com/spmallick/learnopencv/tree/master/Real_Time_Deep_SORT_using_Torchvision_Detectors (https://www.youtube.com/watch?v=GkZRKaQZ_ys)
2) https://www.youtube.com/watch?v=jIRRuGN0j5E
3) https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking/tree/main (https://www.youtube.com/watch?v=9jRRZ-WL698)
4) https://github.com/RizwanMunawar/yolov8-object-tracking
5) https://www.youtube.com/watch?v=jvgmnJspjoA
