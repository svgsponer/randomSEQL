#!/bin/bash

coverage run -m unittest discover test -v
coverage report -m
