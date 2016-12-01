export MXNET_ROOT=`pwd`/..
# Change this to path of openblas
ifndef OPENBLAS_ROOT
	export OPENBLAS_ROOT=${MXNET_ROOT}/../OpenBLAS
endif

ifndef PREFIX
	export PREFIX=${MXNET_ROOT}/lib
endif

# Whether use minimum build without blas and SSE, this will make the library super slow
ifndef MIN
	export MIN=0
endif

ifndef ANDROID
    export ANDROID=0
endif


.PHONY: all clean


CFLAGS=-std=c++11 -Wno-unknown-pragmas -Wall
ifneq ($(MIN), 1)
	CFLAGS += -I${OPENBLAS_ROOT} -I${OPENBLAS_ROOT}/include
	LDFLAGS+= -L${OPENBLAS_ROOT} -L${OPENBLAS_ROOT}/lib -lopenblas
endif


all: android libmxnet_predict.a ${PREFIX}/libmxnet_predict.so

mxnet_predict0.d: mxnet_predict0.cc
	${CXX} ${CFLAGS} -MD -MF $@ \
	-I ${MXNET_ROOT}/ -I ${MXNET_ROOT}/mshadow/ -I ${MXNET_ROOT}/dmlc-core/include \
	-I ${MXNET_ROOT}/include \
	-D__MIN__=$(MIN) -c $+
	rm mxnet_predict0.o

mxnet_predict-all.cc:  mxnet_predict0.d mxnet_predict0.cc
	@echo "Generating amalgamation to " $@
	python ./amalgamation.py $+ $@ $(MIN) $(ANDROID)

mxnet_predict-all.o: mxnet_predict-all.cc
	${CXX} ${CFLAGS} -fPIC -o $@ -c $+

libmxnet_predict.a: mxnet_predict-all.o
	ar rcs libmxnet_predict.a $+

jni_libmxnet_predict.o: mxnet_predict-all.cc
	${CXX} ${CFLAGS} -fPIC -o $@ -c jni/predictor.cc

jni_libmxnet_predict.so: jni_libmxnet_predict.o
	${CXX} ${CFLAGS} -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

ifneq ($(ANDROID), 1)
        LDFLAGS+= -lrt
        android:
else
  ifneq ($(BINARY), 64)
    CFLAGS+=  -mhard-float -D_NDK_MATH_NO_SOFTFP=1 -O3
    LDFLAGS+=  -Wl,--no-warn-mismatch -lm_hard
  else
    CFLAGS+=  -D_NDK_MATH_NO_SOFTFP=1 -O3
    LDFLAGS+=  -Wl,--no-warn-mismatch
  endif
  android: jni_libmxnet_predict.so
endif

libmxnet_predict.js: mxnet_predict-all.cc
	emcc -std=c++11 -O2 -D__MXNET_JS__  -o $@ $+ \
	-s EXPORTED_FUNCTIONS="['_MXPredCreate', '_MXPredGetOutputShape', '_MXPredSetInput', '_MXPredForward', '_MXPredPartialForward', '_MXPredGetOutput', '_MXPredFree', '_MXNDListCreate', '_MXNDListGet', '_MXNDListFree']" \
	-s ALLOW_MEMORY_GROWTH=1


${PREFIX}/libmxnet_predict.so:  mxnet_predict-all.o
	@mkdir -p ${PREFIX}
	${CXX} ${CFLAGS} -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)
	ls -alh $@

clean:
	rm -f *.d *.o *.so *.a mxnet_predict-all.cc
