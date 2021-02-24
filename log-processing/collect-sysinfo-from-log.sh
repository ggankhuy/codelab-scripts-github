# Due to log containing both gim/libgv, the scripts need to support both.
# Collect VBIOS information
# Collect how many number of times, gim / libgv is initialized + date and time.
# Collect known error lines + date and time.
# Collect gim /libgv version
# Collect gpu bdf in each initialization.

# VBIOS pattern:
# libgv: <ATOM BIOS:> (newer libgv older libgv may not have, older libgv add'ly can use: <build num:/part info:/build date:>
# gim:  <build num:/part Info:/build date:>

# Initialization starting line: 
# libgv: <Start AMD open source GIM initialization>
# gim: <2019 Q3 Production Release, Build [0-9a-z]>

# Initialization complete line:
# libgv: <AMD GIM is Running>
# gim: <Running GIM>

# gpu initialization line: 
# libgv: <AMD GIM start to probe device>
# gim: <\) found:>

# gpu initialization complete line:
# libgv: <AMD GIM probed GPU>
# gim: <AMD GIM probe: pfCount>

# Common errors:
# <>


