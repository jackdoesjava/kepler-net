.PHONY: format train clean

format:
	ruff check --fix .
	ruff format .

train:
	python -m src.kepler_net.training.trainer

clean:
	Remove-Item -Recurse -Force data\processed\*.npz