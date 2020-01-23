from acquisitions import import_acquisition
from documents import import_documents
from keywords import import_keywords
from sentences import import_sentences

from api.db import Acquisition
from api.db import Sentence
from api.db import Keyword

try:
    import_acquisition()
    import_documents()
    import_sentences()
    import_keywords()

except Exception as err:
    Acquisition.delete_many({})
    Sentence.delete_many({})
    Keyword.delete_many({})
    print("all documents deleted after the following exception:")
    raise err

else:
    print("documents imported from excel successfully")
