import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: card
    property alias title: titleLabel.text
    property color accent: "#3eb29b"

    width: parent ? parent.width : 300
    height: 100
    color: "#1e1e1e"
    radius: 8

    ColumnLayout {
        anchors.centerIn: parent
        spacing: 4
        Label {
            id: titleLabel
            text: ""
            color: "white"
            font.pixelSize: 14
            horizontalAlignment: Text.AlignHCenter
        }
        Rectangle {
            width: parent.width * 0.8
            height: 4
            radius: 2
            color: card.accent
            opacity: 0.3
        }
    }
}
