# API Reference

## POST /predict
Request:
```json
{"Area":1200,"Bedrooms":3,"Bathrooms":2,"Age":5,"Location":"City Center","Property_Type":"Apartment"}
```
Success: `200` with `predicted_price` and `currency`.  
Validation failure: `400` with an `error` message.
