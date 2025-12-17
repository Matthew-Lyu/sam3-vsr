首先介绍一下cambrains的vsr任务

VSR (longhorizon visual spatial recall)

![image-20251217174353671](/Users/lyuweichen/Desktop/sam3/assets/image-20251217174353671.png)

这个任务设计的非常巧妙，很好考验视觉语言模型对于空间真正的感知能力，我们想做一个跟Cambrian-S可以打一打的模型，现在我们处于问题的第一步，首先可以了解一下数据的构成

```bash
(base) xc@duan146:~/Spatial$ cd /data2
(base) xc@duan146:/data2$ ls
cwl  dmy  lab  lwc  xc
(base) xc@duan146:/data2$ cd xc
(base) xc@duan146:/data2/xc$ ls
checkpoints  datasets  eval_logs  hf_cache  models
(base) xc@duan146:/data2/xc$ cd hf_cache/
(base) xc@duan146:/data2/xc/hf_cache$ ls
cambrians_vsc  cambrians_vsr  datasets  egoschema  gradio  hub  vsibench
(base) xc@duan146:/data2/xc/hf_cache$ cd cambrians_vsr
(base) xc@duan146:/data2/xc/hf_cache/cambrians_vsr$ ls
10mins  120mins  240mins  30mins  60mins
(base) xc@duan146:/data2/xc/hf_cache/cambrians_vsr$ cd 10mins
(base) xc@duan146:/data2/xc/hf_cache/cambrians_vsr/10mins$ ls
00000000.mp4  00000004.mp4  00000008.mp4  00000012.mp4  00000016.mp4  00000020.mp4  00000024.mp4  00000028.mp4  00000032.mp4  00000036.mp4  00000040.mp4  00000044.mp4  00000048.mp4  00000052.mp4  00000056.mp4
00000001.mp4  00000005.mp4  00000009.mp4  00000013.mp4  00000017.mp4  00000021.mp4  00000025.mp4  00000029.mp4  00000033.mp4  00000037.mp4  00000041.mp4  00000045.mp4  00000049.mp4  00000053.mp4  00000057.mp4
00000002.mp4  00000006.mp4  00000010.mp4  00000014.mp4  00000018.mp4  00000022.mp4  00000026.mp4  00000030.mp4  00000034.mp4  00000038.mp4  00000042.mp4  00000046.mp4  00000050.mp4  00000054.mp4  00000058.mp4
00000003.mp4  00000007.mp4  00000011.mp4  00000015.mp4  00000019.mp4  00000023.mp4  00000027.mp4  00000031.mp4  00000035.mp4  00000039.mp4  00000043.mp4  00000047.mp4  00000051.mp4  00000055.mp4  00000059.mp4
```

可以看到cambrians_vsr这个数据集包含了10-60分钟的视频，这个长度的视频是长视频，一般的视觉语言模型对于长视频的处理能力有限，因为token太长，所以我们在主模型之前考虑用sam3对视频进行一个预分割，把我们要找的目标物体，比如说上图（ Figure 4）中要找泰迪熊，我们希望先用sam3把熊出现的所有帧都找出来，维护一个json文件，这就是我们要做的第一个任务。
