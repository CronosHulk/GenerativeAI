from langchain_core.documents import Document


POLICIES = [
    Document(page_content="Email is for business communication. Never send passwords or confidential data by email.", metadata={"section": "email"}),
    Document(page_content="Smoking and vaping are forbidden inside offices. Use the designated outdoor smoking area.", metadata={"section": "safety"}),
    Document(page_content="Employees may work remotely two days per week after manager approval.", metadata={"section": "remote work"}),
    Document(page_content="Company devices must use disk encryption and automatic screen locking.", metadata={"section": "security"}),
    Document(page_content="Annual leave requests should be submitted at least two weeks in advance.", metadata={"section": "leave"}),
]

MOVIES = [
    Document(page_content="Scientists bring dinosaurs back and chaos follows.", metadata={"title": "Jurassic Park", "year": 1993, "rating": 7.7, "genre": "science fiction", "director": "Steven Spielberg"}),
    Document(page_content="A thief enters layered dreams to plant an idea.", metadata={"title": "Inception", "year": 2010, "rating": 8.2, "genre": "science fiction", "director": "Christopher Nolan"}),
    Document(page_content="A detective becomes trapped in dreams that alter reality.", metadata={"title": "Paprika", "year": 2006, "rating": 8.6, "genre": "animated", "director": "Satoshi Kon"}),
    Document(page_content="Three men travel through the mysterious Zone.", metadata={"title": "Stalker", "year": 1979, "rating": 9.0, "genre": "science fiction", "director": "Andrei Tarkovsky"}),
    Document(page_content="Four sisters grow up and choose their own paths.", metadata={"title": "Little Women", "year": 2019, "rating": 8.3, "genre": "drama", "director": "Greta Gerwig"}),
]

PARENT_TEXTS = [
    Document(page_content="Workplace safety policy. Smoking and vaping are forbidden in every indoor company area. Employees who smoke must use the marked outdoor zone behind the west parking lot. Repeated violations are reported to HR. Fire exits must remain unobstructed at all times.", metadata={"title": "Safety handbook"}),
    Document(page_content="Information security policy. Every laptop must use full-disk encryption. Screens lock automatically after five minutes. Passwords must never be shared by email or chat. Suspected phishing messages should be forwarded to the security team.", metadata={"title": "Security handbook"}),
]
