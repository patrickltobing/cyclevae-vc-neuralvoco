PYTHON_VERSION_MIN=3.6
PYTHON_VERSION=$(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
PYTHON_VERSION_OK=$(shell python -c 'import sys; print(int(float(${PYTHON_VERSION}) >= ${PYTHON_VERSION_MIN}))')

ifeq ($(PYTHON_VERSION_OK),0)
  $(error "Need python $(PYTHON_VERSION) >= $(PYTHON_VERSION_MIN)")
endif
PYTHON_VERSION_WITHOUT_DOT = $(strip $(subst .,,$(PYTHON_VERSION)))

CUDA_VERSION := 10.1
CUDA_VERSION_WITHOUT_DOT = $(strip $(subst .,,$(CUDA_VERSION)))

PYTORCH_VERSION := 1.7.0
TORCHVISION_VERSION := 0.8.1

.PHONY: all clean

all: activate torch

activate:
	test -d venv || virtualenv -p python$(PYTHON_VERSION) venv
	. venv/bin/activate; pip install pip --upgrade; pip install -r requirements.txt; pip install matplotlib;
	touch venv/bin/activate

torch: activate
	. venv/bin/activate; pip install pip --upgrade; \
        pip3 install https://download.pytorch.org/whl/cu$(CUDA_VERSION_WITHOUT_DOT)/torch-$(PYTORCH_VERSION)%2Bcu$(CUDA_VERSION_WITHOUT_DOT)-cp$(PYTHON_VERSION_WITHOUT_DOT)-cp$(PYTHON_VERSION_WITHOUT_DOT)m-linux_x86_64.whl; pip3 install torch-optimizer; \
            pip3 install https://download.pytorch.org/whl/cu$(CUDA_VERSION_WITHOUT_DOT)/torchvision-$(TORCHVISION_VERSION)%2Bcu$(CUDA_VERSION_WITHOUT_DOT)-cp$(PYTHON_VERSION_WITHOUT_DOT)-cp$(PYTHON_VERSION_WITHOUT_DOT)m-linux_x86_64.whl

clean:
	rm -fr venv
	find -iname "*.pyc" -delete
