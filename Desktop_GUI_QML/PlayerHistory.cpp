#include "PlayerHistory.h"
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>

PlayerHistory::PlayerHistory(QObject *parent) : QAbstractListModel(parent)
{
    load();
}

int PlayerHistory::rowCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return 0;
    return m_players.count();
}

QVariant PlayerHistory::data(const QModelIndex &index, int role) const
{
    if (!index.isValid() || index.row() < 0 || index.row() >= m_players.size())
        return QVariant();

    if (role == NameRole || role == Qt::DisplayRole)
        return m_players.at(index.row());

    return QVariant();
}

void PlayerHistory::addPlayer(const QString &name)
{
    QString trimmed = name.trimmed();
    if (trimmed.isEmpty())
        return;

    int pos = m_players.indexOf(trimmed);
    if (pos != -1)
    {
        if (pos == 0)
            return; // already first
        beginMoveRows(QModelIndex(), pos, pos, QModelIndex(), 0);
        m_players.move(pos, 0);
        endMoveRows();
    }
    else
    {
        beginInsertRows(QModelIndex(), m_players.size(), m_players.size());
        m_players.append(trimmed);
        endInsertRows();
    }
    save();
}

QStringList PlayerHistory::players() const
{
    return m_players;
}

void PlayerHistory::load()
{
    QFile file("player_history.json");
    if (!file.open(QIODevice::ReadOnly))
        return;

    QByteArray data = file.readAll();
    file.close();

    auto doc = QJsonDocument::fromJson(data);
    if (!doc.isArray())
        return;

    beginResetModel();
    m_players.clear();
    for (const QJsonValue &v : doc.array())
    {
        if (v.isString())
            m_players.append(v.toString());
    }
    endResetModel();
}

void PlayerHistory::save() const
{
    QFile file("player_history.json");
    if (!file.open(QIODevice::WriteOnly))
        return;

    QJsonArray arr;
    for (const QString &name : m_players)
        arr.append(name);
    QJsonDocument doc(arr);
    file.write(doc.toJson());
    file.close();
}

QHash<int, QByteArray> PlayerHistory::roleNames() const
{
    return {{NameRole, "name"}};
}

