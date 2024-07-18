#!/bin/bash
pipenv run tensorflowjs_converter --input_format keras alexnet_model.h5 ./web/tfjs_model
