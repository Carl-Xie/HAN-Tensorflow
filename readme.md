An implementation of Hierarchical Attention Networks,
according to this paper https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
The following picture is the model architecture:
![](https://github.com/Carl-Xie/HAN-Tensorflow/blob/master/han.png)

This implementation is more about learning how to use TensorFlow to build and train a model,
specifically, you will learn how to:
    - build a model in a way that is readable and maintainable
    - use tensorflow's reading pipeline to read data from file
    - train and test a model
    - write summaries and visualize with tensorboard
    - save and restore a model from checkpoint file
It assumes you know the basic operations in tensorflow, if not, you should
check the [official guide](https://www.tensorflow.org/) first.

The model's hyper-parameters is not being tuned, and the small dataset is extracted
from
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
and is used for sanity check. You will see the model overfitting quickly within 10 epochs.

