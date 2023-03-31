# Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics
![Network Architecture](Network.PNG)


**About**

This is the source code for the paper 

Chen Li, Zhen Zhang, Wee Sun Lee, Gim Hee Lee. Convolutional Sequence to Sequence Model for Human Dynamics. In CVPR2018.

The paper proposes a Convolotinal sequence-to-sequence model for human motion prediction. For more details, please refer to our paper on arxiv: http://arxiv.org/abs/1805.00655.pdf.

Bibtex:
```
@inproceedings{li2018convolutional,
  title={Convolutional sequence to sequence model for human dynamics},
  author={Li, Chen and Zhang, Zhen and Lee, Wee Sun and Lee, Gim Hee},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5226--5234},
  year={2018}
}
```

**Dependencies**
1. h5py--to save samples
2. Tensorflow 1.2 or later

**Train**

Get this code:
```
git clone https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics.git
```
Run:
```
python src/AC_main.py 
```
You can also change the arguments during training, for example,you can change the training dataset to CMU dataset by run:
```
python src/AC_main.py --dataset cmu
```
You can also change other arguments in the AC_main.py in a similar way.

**Test**

Run:
```
python src/AC_main.py --is_sampling True --checkpoint 24000 (if you are using the pretrained model)
```
You will specify 'checkpoint' to other value if you use your own model. This command will also generate the sample file which you can use for visualization.  

 **Visualize**
 
To visualize the predicted results, run:
```
python src/forward_kinematics.py (you have to specify the sample file in the code)
```
or run:
```
python src/forward_kinematics_cmu.py (for the CMU dataset)
```

**Acknowledgments**

The pre-processed human3.6 dataset and some of our evaludation code was ported or adapted from SRNN [@asheshjain399](https://github.com/asheshjain399/RNNexp) and RRNN by [@una-dinosauria](https://github.com/una-dinosauria/human-motion-prediction).
