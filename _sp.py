import wbgr_datawarehouse.db as dwh
import wbgr_datawarehouse.tables as tables


db = dwh.db()
session = db.get_session()
tag_name = '\\PRD-SPH-002\SRV-ABB-DP\800xA\PDP-O-PT00-FQT01.Status.GebruikActueel.Value'
query = session\
    .query(tables.dim_signaal)\
    .filter(tables.dim_signaal.dsg_tag == repr(tag_name)[1:-1])\
    .first()
print(query)
print(tag_name)
print(repr(tag_name))
print(repr(tag_name)[1:-1])
