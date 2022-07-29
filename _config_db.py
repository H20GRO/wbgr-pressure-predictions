from typing import List
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, scoped_session, sessionmaker,Session

base = declarative_base()
engine = sqlalchemy.create_engine('sqlite:///sqlalchemy.sqlite', echo=True)
sessionmaker = sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=engine)
def get_session() -> Session:
    with sessionmaker() as session:
        return session

class Nodes(base):
    """e.g. DMA"""
    __tablename__ = "Nodes"
    NodeId = sqlalchemy.Column(sqlalchemy.INTEGER, primary_key=True)
    Name = sqlalchemy.Column(sqlalchemy.VARCHAR(200), unique=True)
    Sources = relationship('Connections',foreign_keys='[Connections.SourceId]', back_populates='Source')
    Destinations = relationship('Connections',foreign_keys='[Connections.DestinationId]',back_populates='Destination')

class Signals(base):
    """Signals with SPH name referring to the data underlying"""
    __tablename__ = "Signals"
    SignalId = sqlalchemy.Column(sqlalchemy.INTEGER, primary_key=True)
    Name = sqlalchemy.Column(sqlalchemy.VARCHAR(200),unique=True)
    SPHTagName = sqlalchemy.Column(sqlalchemy.VARCHAR(200))
    Connections = relationship('Connections',foreign_keys='[Connections.SignalId]', back_populates='Signal')

class Connections(base):
    """Routing to Nodes """
    __tablename__ = "Connections"
    __table_args__ = (            
            sqlalchemy.UniqueConstraint('SourceId', 'DestinationId',name='UC_SOURCE_DESTINATION'),            
            )
    ConnectionId = sqlalchemy.Column(sqlalchemy.INTEGER, primary_key=True)
    Name = sqlalchemy.Column(sqlalchemy.VARCHAR(200))
    SourceId = sqlalchemy.Column(sqlalchemy.ForeignKey(Nodes.NodeId))
    Source = relationship('Nodes',foreign_keys=[SourceId])
    DestinationId = sqlalchemy.Column(sqlalchemy.ForeignKey(Nodes.NodeId))
    Destination = relationship('Nodes',foreign_keys=[DestinationId])
    SignalId = sqlalchemy.Column(sqlalchemy.ForeignKey(Signals.SignalId))
    Signal = relationship('Signals',foreign_keys=[SignalId])

if __name__ == '__main__':
    session = get_session()
    regenerate = False
    if regenerate:            
        base.metadata.drop_all(engine)
        base.metadata.create_all(engine)
        
    nodes = ['Provincie','Stad','PON','PDP-O','PDP-G','ANN','ONO','ORU','DEE','DOO','']
    signals = ['OnnenFlow','DePuntOW','Ruischerbrug']

    for nodeName in nodes:
        if session.query(Nodes.Name).filter(Nodes.Name == nodeName).first():
            continue
        node = Nodes()
        node.Name = nodeName
        session.add(node)    
    session.commit()

    for signalName in signals:
        if session.query(Signals).filter(Signals.Name == signalName).first():
            continue
        signal = Signals()
        signal.Name = signalName
        signal.SPHTagName = f"SPH:{signalName}"
        session.add(signal)    
    session.commit()

    nodes = session.query(Nodes).all()
    signals = session.query(Signals).all()
    conn = Connections()
    conn.Destination = nodes[0]
    conn.Source = nodes[1]
    conn.Signal = signals[1]
    conn.Name = f"From {conn.Source.Name} to  {conn.Destination.Name}"
    session.add(conn)
    session.commit()
    
    


    


