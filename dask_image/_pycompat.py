# -*- coding: utf-8 -*-

try:
    irange = xrange
except NameError:
    irange = range

try:
    from itertools import imap, izip
except ImportError:
    imap, izip = map, zip

try:
    unicode = unicode
except NameError:
    unicode = str

strlike = (bytes, unicode)
