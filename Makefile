PROJECT_NAME := ml-mgie

.PHONY: venv
venv:
	python -m venv venv_${PROJECT_NAME} && echo "run: source venv_${PROJECT_NAME}/bin/activate"

.PHONY: check_type
check_type: ## run mypy
	python -m mypy --config-file=./mypy.ini ./${PROJECT_NAME}/${PROJECT_NAME}

.PHONY: tests
tests: check_type
	python -u -m pytest ./${PROJECT_NAME}/tests -vv -s

