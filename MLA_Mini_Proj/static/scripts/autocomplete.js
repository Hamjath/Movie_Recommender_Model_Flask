// static/scripts/autocomplete.js
document.addEventListener("DOMContentLoaded", function () {
    const input = document.getElementById("movie-input");
    const list = document.getElementById("autocomplete-list");
    let currentRequest = null;
    let selectedIndex = -1;

    function clearList() {
        list.innerHTML = "";
        selectedIndex = -1;
    }

    function makeItem(text) {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "list-group-item list-group-item-action";
        item.textContent = text;
        item.addEventListener("click", () => {
            input.value = text;
            clearList();
            document.getElementById("search-form").submit();
        });
        return item;
    }

    input.addEventListener("input", function () {
        const q = input.value.trim();
        clearList();
        if (q.length < 2) return;

        // Abort previous request if still pending
        if (currentRequest) currentRequest.abort && currentRequest.abort();

        // Use fetch with a simple timeout/abort via AbortController
        const controller = new AbortController();
        currentRequest = controller;

        fetch(`/api/suggest?q=${encodeURIComponent(q)}`, { signal: controller.signal })
            .then(resp => resp.json())
            .then(data => {
                clearList();
                if (!Array.isArray(data) || data.length === 0) return;
                data.forEach(title => {
                    list.appendChild(makeItem(title));
                });
            })
            .catch(err => {
                // ignore aborted or network errors silently
            })
            .finally(() => {
                currentRequest = null;
            });
    });

    // basic keyboard navigation
    input.addEventListener("keydown", function (e) {
        const items = list.querySelectorAll(".list-group-item");
        if (!items.length) return;
        if (e.key === "ArrowDown") {
            e.preventDefault();
            selectedIndex = (selectedIndex + 1) % items.length;
            items.forEach((it, i) => it.classList.toggle("active", i === selectedIndex));
            items[selectedIndex].scrollIntoView({ block: "nearest" });
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            selectedIndex = (selectedIndex - 1 + items.length) % items.length;
            items.forEach((it, i) => it.classList.toggle("active", i === selectedIndex));
            items[selectedIndex].scrollIntoView({ block: "nearest" });
        } else if (e.key === "Enter") {
            if (selectedIndex >= 0 && items[selectedIndex]) {
                e.preventDefault();
                items[selectedIndex].click();
            }
        }
    });

    // click away to close
    document.addEventListener("click", function (e) {
        if (!list.contains(e.target) && e.target !== input) clearList();
    });
});
