#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import

import dask_image._pycompat


def test_irange():
    r = dask_image._pycompat.irange(5)

    assert not isinstance(r, list)

    assert list(r) == [0, 1, 2, 3, 4]


def test_imap():
    r = dask_image._pycompat.imap(lambda e : e ** 2, [0, 1, 2, 3])

    assert not isinstance(r, list)

    assert list(r) == [0, 1, 4, 9]


def test_izip():
    r = dask_image._pycompat.izip([1, 2], [3, 4, 5])

    assert not isinstance(r, list)

    assert list(r) == [(1, 3), (2, 4)]


def test_strlike():
    b = b"Hello World!"
    u = u"Hello World!"

    assert isinstance(b, dask_image._pycompat.strlike)
    assert isinstance(u, dask_image._pycompat.strlike)


def test_unicode():
    u = u"Hello World!"

    assert isinstance(u, dask_image._pycompat.unicode)
