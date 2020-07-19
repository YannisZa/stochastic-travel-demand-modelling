# Stochastic modelling of urban travel demand

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

This repository contains code for Bayesian calibration of the deterrministic Harris Wilson ordinary differential equation and its stochastic extension. The differential equations are defined based on a singly or doubly constrained spatial interaction model of urban transport. The code was based on this [repository](https://github.com/lellam/cities_and_regions) and was created for the purposes of my MRes thesis at the University of Cambridge. My thesis will be released soon on my [website](https://yannisza.github.io/).

Before running any python script please run ```chmod +x ./setup.sh ``` and then ``` ./setup.sh ```. This bash script installs python requirements, sets up necessary directories etc.

This repository is a work in progress. The following tasks have to be completed
- [ ] Make doubly constrained spatial interaction model class compatible to the singly constrained model class.
- [ ] Complete `generate_figures.sh` script.
- [ ] Improve comments on model classes and argsparse descriptions in inference classes.
- [ ] Write tests for all classes.
- [ ] Release outputs in nice format.

More updates soon!

Give the repository a star and fork it if you like it!
