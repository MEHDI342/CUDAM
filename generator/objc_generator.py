// Contrôle d'accès vulnérable (JavaScript)
app.get('/api/user/:id/data', function(req, res) {
    db.getUserData(req.params.id).then(data => {
        res.json(data);  // Pas de vérification d'autorisation
});
});

// Solution sécurisée
app.get('/api/user/:id/data', authorize, function(req, res) {
if (req.user.id !== req.params.id && !req.user.isAdmin) {
return res.status(403).json({error: 'Accès non autorisé'});
}
db.getUserData(req.params.id).then(data => {
    res.json(data);
});
});
