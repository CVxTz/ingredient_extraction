install:
	pip install -e .[dev]
test:
	python -m pytest tests

format:
	ruff format . && ruff check --fix .

lint:
	#disable comment to test speed
	#pylint --disable=R,C --ignore-patterns=test_.*?py *.py mylib/*.py
	#ruff linting is 10-100X faster than pylint
	ruff check *.py m