Use the makefile for most actions:

* Build: `maturin develop`
* Test: `make test`
* Run single test: `pytest python/tests/<test_file>.py::<test_name>`
* Doctest: `make doctest`
* Lint: `make lint`
* Format: `make format`


If you want to run python tests after changes to the rust code, you need first build the rust code by:
```
maturin develop
```
