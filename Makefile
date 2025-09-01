.PHONY: help install lint test clean all

# Default target
help:
	@echo "Available commands:"
	@echo "  install     Install all dependencies"
	@echo "  lint        Run all linting checks"
	@echo "  test        Run tests with coverage"
	@echo "  clean       Clean up generated files"
	@echo "  all         Run lint and test"

# Install dependencies
install:
	pip3 install --upgrade pip
	pip3 install -r requirements.txt

# Run all linting checks
lint:
	@echo "Running flake8..."
	flake8 src/
	@echo "✅ All linting checks passed!"

# Run tests
test:
	@echo "Running tests with pytest..."
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "✅ Tests completed! Coverage report available in htmlcov/"

# Clean up generated files
clean:
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf test-results/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

# Run everything
all: lint test