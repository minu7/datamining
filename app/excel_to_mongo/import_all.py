from acquisitions import import_acquisition
from documents import import_documents
from keywords import import_keywords
from paragraphs import import_paragraphs

from api.db import Acquisition
from api.db import Paragraph
from api.db import Keyword

try:
    import_acquisition()
    import_documents()
    import_paragraphs()
    import_keywords()

except Exception as err:
    Acquisition.delete_many({})
    Paragraph.delete_many({})
    Keyword.delete_many({})
    print("all documents deleted after the following exception:")
    raise err

else:
    print("documents imported from excel successfully")
