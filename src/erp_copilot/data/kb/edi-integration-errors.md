# EDI and Integration Errors

## Rejected purchase orders
An EDI purchase order is rejected with code 850-REJ when the item number is unknown or the unit of measure does not match the item master. Correct the item mapping and resend the document.

## Retries
Failed integration messages are retried automatically three times at 15 minute intervals. After the third failure the message moves to the error queue and an integration engineer is notified.
