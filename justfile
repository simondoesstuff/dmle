_default:
	@just --list

mlp *args="":
	uv run god-mlp {{args}}

mlp-grokk *args="":
	uv run god-mlp-grokk {{args}}

pcn *args="":
	uv run god-pcn {{args}}

typecheck path='src':
	uv run basedpyright {{path}}

alias t := test
test *args:
	just typecheck && echo ''
	uv run pytest {{args}}

alias i := install
install:
	uv sync
