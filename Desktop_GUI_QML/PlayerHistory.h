#ifndef PLAYERHISTORY_H
#define PLAYERHISTORY_H

#include <QAbstractListModel>
#include <QStringList>

class PlayerHistory : public QAbstractListModel
{
    Q_OBJECT
public:
    enum Roles {
        NameRole = Qt::UserRole + 1
    };

    explicit PlayerHistory(QObject *parent = nullptr);

    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;

    Q_INVOKABLE void addPlayer(const QString &name);
    Q_INVOKABLE QStringList players() const;

    void load();
    void save() const;

protected:
    QHash<int, QByteArray> roleNames() const override;

private:
    QStringList m_players;
};

#endif // PLAYERHISTORY_H
