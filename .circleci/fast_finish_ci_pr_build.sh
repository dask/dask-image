#!/bin/bash

python .gen_ci_support/ff_ci_pr_build.py -v \
	--ci "circle" \
	"${CIRCLE_PROJECT_USERNAME}/${CIRCLE_PROJECT_REPONAME}" \
	"${CIRCLE_BUILD_NUM}" \
	"${CIRCLE_PR_NUMBER}"
