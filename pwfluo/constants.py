import locale

NO_INDEX = 0  # This is the index value for single images
if locale.getencoding() == "UTF-8":
    MICRON_STR = "μm"
else:
    MICRON_STR = "um"
