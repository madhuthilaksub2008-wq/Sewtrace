"""Pixel size of a PNG or JPEG from its header, without decoding.

A small file can declare a huge image (a PNG of one gigapixel compresses to a few MB), and
decoding it would use all the memory of a small host. So the size is read first."""

from __future__ import annotations

import struct

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_NO_LENGTH = {0x01, *range(0xD0, 0xD8)}  # TEM and RSTn: markers without a length field
_NOT_FRAME = {0xC4, 0xC8, 0xCC}  # DHT, JPG, DAC share the SOFn range but are not frame headers
# Markers that carry a length: SOF0..SOF15, DHT, DAC, DQT, DNL, DRI, DHP, EXP, APPn, JPGn, COM.
# Any other byte after FF (00, 02..BF, SOI, EOI, SOS) is refused. libjpeg scans past FF 00 and
# other reserved codes as garbage, so reading them as segments would show a different frame than
# the decoder uses (a fake small size hidden inside a segment).
_WITH_LENGTH = {*range(0xC0, 0xD0), *range(0xDB, 0xFF)}


def image_size(data: bytes) -> tuple[int, int] | None:
    """``(width, height)`` from the PNG IHDR or the JPEG frame header, or None when the header
    is missing or cut off."""
    if data[:8] == PNG_SIGNATURE:
        if len(data) < 24 or data[12:16] != b"IHDR":
            return None
        width, height = struct.unpack(">II", data[16:24])
        return (width, height) if width and height else None
    if data[:3] == b"\xff\xd8\xff":
        return _jpeg_size(data)
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Size from the first frame header, or None for anything a decoder could read differently."""
    pos = 2
    while pos + 2 <= len(data):
        if data[pos] != 0xFF:
            return None
        marker = data[pos + 1]
        if marker == 0xFF:  # fill byte before a marker
            pos += 1
            continue
        if marker in _NO_LENGTH:
            pos += 2
            continue
        if marker not in _WITH_LENGTH or pos + 4 > len(data):
            return None
        (length,) = struct.unpack(">H", data[pos + 2:pos + 4])
        if length < 2:
            return None
        if 0xC0 <= marker <= 0xCF and marker not in _NOT_FRAME:
            if length < 8 or pos + 9 > len(data):
                return None
            height, width = struct.unpack(">HH", data[pos + 5:pos + 9])
            return (width, height) if width and height else None
        pos += 2 + length
    return None
