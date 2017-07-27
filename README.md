The original repository url: https://github.com/jonathanmasci/ShapeNet

## my demo

sudo python queue/streamer_device.py --queue_size 2 &
sudo python queue/streamer_device.py --queue_size 2 --inport 5581 --outport 5582 &

sudo THEANO_FLAGS='device=gpu1,optimizer_including=cudnn,cuda.root=/usr/local/cuda' python lscnn_producer.py --desc_dir /home/chenzhixing/data/BU3D/submesh/descs --shape_dir /home/chenzhixing/data/BU3D/submesh/shapes/ --disk_dir /home/chenzhixing/data/BU3D/submesh/disks/ --const_fun 0 --alltomem 0 --batch_size -1 --queue_size 5 --nthreads 1 --mode train --prm_i 0 --cross_x 10 &
sudo THEANO_FLAGS='device=gpu1,optimizer_including=cudnn,cuda.root=/usr/local/cuda' python lscnn_producer.py --desc_dir /home/chenzhixing/data/BU3D/submesh/descs --shape_dir /home/chenzhixing/data/BU3D/submesh/shapes/ --disk_dir /home/chenzhixing/data/BU3D/submesh/disks/ --const_fun 0 --alltomem 0 --batch_size -1 --queue_size 5 --nthreads 1 --mode valid --prm_i 0 --cross_x 10 &

sudo THEANO_FLAGS='device=gpu1,optimizer_including=cudnn,cuda.root=/usr/local/cuda' python train_lscnn.py --config_file configs/gcnn.yaml --mode train --l2reg 0.0 --queue_size 5 &
