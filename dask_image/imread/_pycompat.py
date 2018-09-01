# -*- coding: utf-8 -*-

try:
    irange = xrange
except NameError:
    irange = range

try:
    from itertools import izip
except ImportError:
    izip = zip
