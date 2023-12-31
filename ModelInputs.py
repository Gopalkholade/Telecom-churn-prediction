from pydantic import BaseModel

class ModelInput(BaseModel):
    gender: str
    age: int
    married: str 
    numberofdependents: int
    city: str 
    zipcode: int 
    numberofreferrals: int
    tenureinmonths: int
    offer: str
    phoneservice: str 
    avgmonthlylongdistancecharges: float 
    multiplelines: str 
    internetservice: str
    internettype: str
    avgmonthlygbdownload: float
    onlinesecurity: str
    onlinebackup: str
    deviceprotectionplan: str
    premiumtechsupport: str
    streamingtv: str
    streamingmovies: str
    streamingmusic: str
    unlimiteddata: str
    contract: str
    paperlessbilling: str 
    paymentmethod: str
    monthlycharge: float
    totalcharges: float
    totalrefunds: float
    totalextradatacharges: int
    totallongdistancecharges: float
    totalrevenue: float
