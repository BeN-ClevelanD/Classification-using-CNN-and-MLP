run: install
	. venv/bin/activate

install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt
venv :
	test -d venv || python3 -m venv venv
clean:
	rm -rf venv
	find -iname "*.pyc" -delete

runMLP:
	python3 MLP.py

runCNN:
	python3 CNN.py
