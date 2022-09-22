## Performance

**mIoU** is reported!

The Shape to calculate FLOPs is **(512, 512)**, the device is a single V100!


|                   Configs                   | whole+ss | whole+ms | silde+ss | slide+ms |
|:-------------------------------------------:|:--------:|:--------:|:--------:|:--------:|
| eunet-t_segfhead_2x4_512x512_160k_ade20k.py |  35.33   |  36.71   |  35.66   |  36.52   |


|                    Configs                    |   Test   | mIoU  | FLOPs(G) | Params(M) |
|:---------------------------------------------:|:--------:|:-----:|:--------:|:---------:|
|  eunet-t_segfhead_2x4_512x512_160k_ade20k.py  | whole+ss | 35.33 |          |           |
|  eunet-t_segfhead_2x4_512x512_160k_ade20k.py  | whole+ms | 36.71 |          |           |
|  eunet-t_segfhead_2x4_512x512_160k_ade20k.py  | slide+ss | 35.66 |          |           |
|  eunet-t_segfhead_2x4_512x512_160k_ade20k.py  | slide+ss | 36.52 |          |           |
| eunet-t_segfhead3_2x8_512x512_160k_ade20k.py  | whole+ss | 35.47 |   2.73   |    2.5    |
| eunet-t_segfhead3_2x8_512x512_160k_ade20k.py  | whole+ms | 36.95 |          |           |
| eunet-t_segfhead3_2x8_512x512_160k_ade20k.py  | slide+ss |       |          |           |
| eunet-t_segfhead3_2x8_512x512_160k_ade20k.py  | slide+ms |       |          |           |
| eunet-t_non-local_2x8_512x512_160k_ade20k.py  | whole+ss | 36.53 |   9.99   |   4.22    |
|                                               |          |       |          |           |
| eunet-t_non-local2_2x8_512x512_160k_ade20k.py |          |       |   8.02   |   3.79    |
| eunet-t_non-local2_2x8_512x512_160k_ade20k.py |          |       |          |           |