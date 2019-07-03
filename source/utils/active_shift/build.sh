#cp -r ./lib/cuda /usr/local/lib/python3.5/dist-packages/tensorflow/include/
#sudo ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so


TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

/usr/local/cuda/bin/nvcc -std=c++11 -DNDEBUG -D_MWAITXINTRIN_H_INCLUDED -c -o active_shift2d.cu.o active_shift2d.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
--expt-relaxed-constexpr -gencode arch=compute_75,code=sm_75 

g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o active_shift2d.so active_shift2d.cc \
  active_shift2d.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}






