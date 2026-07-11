"""Format identity constants for the ``.gnos`` file format.

Kept in their own module so both the package ``__init__`` and the storage layer
can import them without circular-import headaches.
"""

FORMAT_NAME = "mindfile"
# Bump FORMAT_VERSION for spec/behaviour changes; SCHEMA_VERSION for table changes.
FORMAT_VERSION = "1.1.0"
SCHEMA_VERSION = 14
EXTENSION = ".gnos"

# In-content magic number (SQLite `PRAGMA application_id`) so a file
# self-identifies as a mindfile regardless of its extension. 'GNOS' as bytes.
APPLICATION_ID = int.from_bytes(b"GNOS", "big")  # 0x474E4F53 = 1196314451
