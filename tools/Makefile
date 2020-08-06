.PHONY: all clean

#all: venv36cu101v151/bin/activate venv36cu101v151/lib/python3.6/site-packages/torch
all: venv37cu100v141/bin/activate venv37cu100v141/lib/python3.7/site-packages/torch
#all: venv37cu101v151/bin/activate venv37cu101v151/lib/python3.7/site-packages/torch

#venv36cu101v151/bin/activate:
	#test -d venv36cu101v151 || virtualenv -p python3.6 venv36cu101v151
	#. venv36cu101v151/bin/activate; pip install -r requirements.txt; pip install matplotlib;
	#touch venv36cu101v151/bin/activate
venv37cu100v141/bin/activate:
	test -d venv37cu100v141 || virtualenv -p python3.7 venv37cu100v141
	. venv37cu100v141/bin/activate; pip install -r requirements.txt; pip install matplotlib;
	touch venv37cu100v141/bin/activate
#venv37cu101v151/bin/activate:
#	test -d venv37cu101v151 || virtualenv -p python3.7 venv37cu101v151
#	. venv37cu101v151/bin/activate; pip install -r requirements.txt; pip install matplotlib;
#	touch venv37cu101v151/bin/activate

#venv36cu101v151/lib/python3.6/site-packages/torch: venv36cu101v151/bin/activate
#	. venv36cu101v151/bin/activate; pip install pip --upgrade; \
#		pip3 install https://download.pytorch.org/whl/cu101/torch-1.5.1%2Bcu101-cp36-cp36m-linux_x86_64.whl; \
#			pip3 install https://download.pytorch.org/whl/cu101/torchvision-0.6.1%2Bcu101-cp36-cp36m-linux_x86_64.whl
venv37cu100v141/lib/python3.7/site-packages/torch: venv37cu100v141/bin/activate
	. venv37cu100v141/bin/activate; pip install pip --upgrade; \
        pip3 install https://download.pytorch.org/whl/cu100/torch-1.4.0%2Bcu100-cp37-cp37m-linux_x86_64.whl; \
            pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.5.0%2Bcu100-cp37-cp37m-linux_x86_64.whl
#venv37cu101v151/lib/python3.7/site-packages/torch: venv37cu101v151/bin/activate
#	. venv37cu101v151/bin/activate; pip install pip --upgrade; \
#		pip3 install https://download.pytorch.org/whl/cu101/torch-1.5.1%2Bcu101-cp37-cp37m-linux_x86_64.whl; \
#			pip3 install https://download.pytorch.org/whl/cu101/torchvision-0.6.1%2Bcu101-cp37-cp37m-linux_x86_64.whl

clean:
	#rm -fr venv36cu101v151
	rm -fr venv37cu100v141
	#rm -fr venv37cu101v151
	find -iname "*.pyc" -delete
