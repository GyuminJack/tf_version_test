## TEST Model running time Tensorflow2 (Subclass VS Sequential API)
DEVICE CPU : Intel(R) Xeon(R) CPU E5-2640 v4
DEVICE GPU : Quadro M6000
Test Algorithm : [DCGAN](https://www.tensorflow.org/tutorials/generative/dcgan?hl=ko)

### Performance
- CPU
    - Sequential & tf.function : 52s
    - Subclass & tf.function : 45s

- GPU
    - Sequential & tf.funtion : 9.1s
    - Sequential & Without tf.function : 21s
    - Subclass & tf.function : 5.75s
    - Subclass & Without tf.function : 20s

- subclass, tf.function 모두 속도차이를 가져옴.
- 가장 빠른것은 두개를 같이 사용하는 것.

### 참고
- subclass에서 tf.function을 사용하기 위해서는 모든 레이어가 컴파일되야 하므로, class의 __init__에 모든 레이어가 선언되어야 함.
