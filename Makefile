SHELL          = /bin/bash
PROJECT_NAME   = envpool
PROJECT_FOLDER = $(PROJECT_NAME) third_party examples benchmark
PYTHON_FILES   = $(shell find . -type f -name "*.py")
CPP_FILES      = $(shell find $(PROJECT_NAME) -type f -name "*.h" -o -name "*.cc")
BAZEL_FILES    = $(shell find . -type f -name "*BUILD" -o -name "*.bzl")
COMMIT_HASH    = $(shell git log -1 --format=%h)
COPYRIGHT      = "Garena Online Private Limited"
BAZELOPT       =
DATE           = $(shell date "+%Y-%m-%d")
# DOCKER_TAG     = $(DATE)-$(COMMIT_HASH)
DOCKER_TAG     = may
DOCKER_USER    = trinkle23897
PATH           := $(HOME)/go/bin:$(PATH)

# installation

check_install = python3 -c "import $(1)" || (cd && pip3 install $(1) --upgrade && cd -)
check_install_extra = python3 -c "import $(1)" || (cd && pip3 install $(2) --upgrade && cd -)

flake8-install:
	$(call check_install, flake8)
	$(call check_install_extra, bugbear, flake8_bugbear)

py-format-install:
	$(call check_install, isort)
	$(call check_install, yapf)

mypy-install:
	$(call check_install, mypy)

cpplint-install:
	$(call check_install, cpplint)

clang-format-install:
	command -v clang-format || sudo apt-get install -y clang-format

clang-tidy-install:
	command -v clang-tidy || sudo apt-get install -y clang-tidy

go-install:
	# requires go >= 1.16
	command -v go || (sudo apt-get install -y golang-1.18 && sudo ln -sf /usr/lib/go-1.18/bin/go /usr/bin/go)

bazel-install: go-install
	command -v bazel || (go install github.com/bazelbuild/bazelisk@latest && ln -sf $(HOME)/go/bin/bazelisk $(HOME)/go/bin/bazel)

buildifier-install: go-install
	command -v buildifier || go install github.com/bazelbuild/buildtools/buildifier@latest

addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

doc-install:
	$(call check_install, pydocstyle)
	$(call check_install_extra, doc8, "doc8<1")
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	$(call check_install_extra, sphinxcontrib.spelling, sphinxcontrib.spelling pyenchant)

auditwheel-install:
	$(call check_install_extra, auditwheel, auditwheel typed-ast patchelf)

# python linter

flake8: flake8-install
	flake8 $(PYTHON_FILES) --count --show-source --statistics

py-format: py-format-install
	isort --check $(PYTHON_FILES) && yapf -r -d $(PYTHON_FILES)

mypy: mypy-install
	mypy $(PROJECT_NAME)

# c++ linter

cpplint: cpplint-install
	cpplint $(CPP_FILES)

clang-format: clang-format-install
	clang-format --style=file -i $(CPP_FILES) -n --Werror

# bazel file linter

buildifier: buildifier-install
	buildifier -r -lint=warn $(BAZEL_FILES)

# bazel build/test

bazel-pip-requirement-dev:
	cd third_party/pip_requirements && (cmp requirements.txt requirements-dev.txt || ln -sf requirements-dev.txt requirements.txt)

bazel-pip-requirement-release:
	cd third_party/pip_requirements && (cmp requirements.txt requirements-release.txt || ln -sf requirements-release.txt requirements.txt)

clang-tidy: clang-tidy-install bazel-pip-requirement-dev
	bazel build $(BAZELOPT) //... --config=clang-tidy --config=test

bazel-debug: bazel-install bazel-pip-requirement-dev
	bazel run $(BAZELOPT) //:setup --config=debug -- bdist_wheel
	mkdir -p dist
	cp bazel-bin/setup.runfiles/$(PROJECT_NAME)/dist/*.whl ./dist

bazel-build: bazel-install bazel-pip-requirement-dev
	bazel run $(BAZELOPT) //:setup --config=test -- bdist_wheel
	mkdir -p dist
	cp bazel-bin/setup.runfiles/$(PROJECT_NAME)/dist/*.whl ./dist

bazel-release: bazel-install bazel-pip-requirement-release
	bazel run $(BAZELOPT) //:setup --config=release -- bdist_wheel
	mkdir -p dist
	cp bazel-bin/setup.runfiles/$(PROJECT_NAME)/dist/*.whl ./dist

bazel-test: bazel-install bazel-pip-requirement-dev
	bazel test --test_output=all $(BAZELOPT) //... --config=test --spawn_strategy=local --color=yes

bazel-clean: bazel-install
	bazel clean --expunge

# documentation

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -l apache -y 2023 -check $(PROJECT_FOLDER)

docstyle: doc-install
	pydocstyle $(PROJECT_NAME) && doc8 docs && cd docs && make html SPHINXOPTS="-W"

doc: doc-install
	cd docs && make html && cd _build/html && python3 -m http.server

spelling: doc-install
	cd docs && make spelling SPHINXOPTS="-W"

doc-clean:
	cd docs && make clean

doc-benchmark:
	pandoc benchmark/README.md --from markdown --to rst -s -o docs/content/benchmark.rst --columns 1000
	cd benchmark && ./plot.py --suffix png && mv *.png ../docs/_static/images/throughput

lint: buildifier flake8 py-format clang-format cpplint clang-tidy mypy docstyle spelling

format: py-format-install clang-format-install buildifier-install addlicense-install
	isort $(PYTHON_FILES)
	yapf -ir $(PYTHON_FILES)
	clang-format -style=file -i $(CPP_FILES)
	buildifier -r -lint=fix $(BAZEL_FILES)
	addlicense -c $(COPYRIGHT) -l apache -y 2023 $(PROJECT_FOLDER)

# Build docker images

docker-ci:
	docker build --network=host -t $(PROJECT_NAME):$(DOCKER_TAG) -f docker/dev.dockerfile .
	echo successfully build docker image with tag $(PROJECT_NAME):$(DOCKER_TAG)

docker-ci-push: docker-ci
	docker tag $(PROJECT_NAME):$(DOCKER_TAG) $(DOCKER_USER)/$(PROJECT_NAME):$(DOCKER_TAG)
	docker push $(DOCKER_USER)/$(PROJECT_NAME):$(DOCKER_TAG)

docker-ci-launch: docker-ci
	docker run --network=host -v /home/ubuntu:/home/github-action --shm-size=4gb -it $(PROJECT_NAME):$(DOCKER_TAG) bash

docker-run:
		docker run --network=host \
		-v $(shell pwd)/../:/app \
		-v $(HOME)/.cache:/root/.cache \
		-v /:/host \
		--shm-size=4gb -it --rm \
		-e DISPLAY=$(DISPLAY) \
		-e QT_X11_NO_MITSHM=1 \
		-e XAUTHORITY=/tmp/.docker.xauth \
		--gpus all \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(shell test -n "$$XAUTHORITY" && echo "$$XAUTHORITY" || echo "/tmp/.Xauthority"):/tmp/.docker.xauth \
		$(PROJECT_NAME):$(DOCKER_TAG) bash

docker-run-mac:
	mkdir -p /tmp/runtime-dir-$(USER)
	chmod 700 /tmp/runtime-dir-$(USER)
	docker run --network=host \
		-v $(shell pwd)/../:/app \
		-v $(HOME)/.cache:/root/.cache \
		-v /:/host \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v /tmp/runtime-dir-$(USER):/tmp/runtime-dir \
		--shm-size=4gb -it --rm \
		-e DISPLAY=host.docker.internal:0 \
		-e QT_X11_NO_MITSHM=1 \
		- gpus all \
		-e XDG_RUNTIME_DIR=/tmp/runtime-dir \
		-e LIBGL_ALWAYS_INDIRECT=1 \
		-e MESA_GL_VERSION_OVERRIDE=3.3 \
		$(PROJECT_NAME):$(DOCKER_TAG) bash

# docker-run-mac:
# 		mkdir -p /tmp/runtime-dir-$(USER) 
# 		chmod 700 /tmp/runtime-dir-$(USER) 
# 		xhost + 127.0.0.1 
# 		docker run --network=host \
# 			-v $(shell pwd)/../:/app \
# 			-v $(HOME)/.cache:/root/.cache \
# 			-v /tmp/.X11-unix:/tmp/.X11-unix \
# 			-v /tmp/runtime-dir-$(USER):/tmp/runtime-dir \
# 			--shm-size=4gb -it --rm \
# 			-e DISPLAY=host.docker.internal:0 \
# 			-e QT_X11_NO_MITSHM=1 \
# 			-e XDG_RUNTIME_DIR=/tmp/runtime-dir \
# 			-e LIBGL_ALWAYS_INDIRECT=1 \
# 			-e MESA_GL_VERSION_OVERRIDE=3.3 \
# 			-e XAUTHORITY=/tmp/.docker.xauth \
# 			$(PROJECT_NAME):$(DOCKER_TAG) bash

docker-dev: docker-ci docker-run-mac 
#  	docker run --network=host -v /:/host -v $(shell pwd)/../:/app -v $(HOME)/.cache:/root/.cache --shm-size=4gb -it $(PROJECT_NAME):$(DOCKER_TAG) bash

# for mainland China
docker-dev-cn:
	docker build --network=host -t $(PROJECT_NAME):$(DOCKER_TAG) -f docker/dev-cn.dockerfile .
	echo successfully build docker image with tag $(PROJECT_NAME):$(DOCKER_TAG)
	docker run --network=host -v /:/host -v $(shell pwd):/app -v $(HOME)/.cache:/root/.cache --shm-size=4gb -it $(PROJECT_NAME):$(DOCKER_TAG) zsh

docker-release:
	docker build --network=host -t $(PROJECT_NAME)-release:$(DOCKER_TAG) -f docker/release.dockerfile .
	echo successfully build docker image with tag $(PROJECT_NAME)-release:$(DOCKER_TAG)

docker-release-push: docker-release
	docker tag $(PROJECT_NAME)-release:$(DOCKER_TAG) $(DOCKER_USER)/$(PROJECT_NAME)-release:$(DOCKER_TAG)
	docker push $(DOCKER_USER)/$(PROJECT_NAME)-release:$(DOCKER_TAG)

docker-release-launch: docker-release
	docker run --network=host -v /:/host -v $(shell pwd):/app -v $(HOME)/.cache:/root/.cache --shm-size=4gb -it $(PROJECT_NAME)-release:$(DOCKER_TAG) zsh

pypi-wheel: auditwheel-install bazel-release
	ls dist/*.whl -Art | tail -n 1 | xargs auditwheel repair --plat manylinux_2_24_x86_64

release-test1:
	cd envpool && python3 make_test.py

release-test2:
	cd examples && python3 make_env.py && python3 env_step.py

release-test: release-test1 release-test2

custom-t: 
	bazel run --config=test //:setup -- bdist_wheel > bazel_build_log.log
	pip3 uninstall envpool -y
	pip3 install bazel-bin/setup.runfiles/envpool/dist/envpool-0.8.4-cp310-cp310-linux_x86_64.whl 
	python3 examples/env_step.py